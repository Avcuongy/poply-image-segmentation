"""
Swin Transformer backbone for Swin-Unet.

Code structure adapted from the official Microsoft Swin Transformer implementation:
https://github.com/microsoft/Swin-Transformer

The default configuration corresponds to Swin-T:
    patch_size = 4
    embed_dim = 96
    depths = (2, 2, 6, 2)
    num_heads = (3, 6, 12, 24)
    window_size = 7

Pretrained checkpoint example:
    swin_tiny_patch4_window7_224.pth

License/source note:
    Microsoft Swin Transformer is released under the MIT License.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    # Compatibility with older timm versions.
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Args:
        x: [B, H, W, C]

    Returns:
        windows: [num_windows * B, window_size, window_size, C]
    """
    B, H, W, C = x.shape

    if H % window_size != 0 or W % window_size != 0:
        raise ValueError(
            f"Feature resolution ({H}, {W}) must be divisible by "
            f"window_size={window_size}."
        )

    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )

    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Reverse window_partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)

        self.register_buffer(
            "relative_position_index",
            relative_position_index,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B_, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                B_,
                N,
                3,
                self.num_heads,
                C // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(N, N, -1)
            .permute(2, 0, 1)
            .contiguous()
        )

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]

            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N,
            )

            attn = attn + mask.unsqueeze(1).unsqueeze(0)

            attn = attn.view(
                -1,
                self.num_heads,
                N,
                N,
            )

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    """Standard W-MSA / SW-MSA Swin Transformer block."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(input_resolution) <= window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must satisfy 0 <= shift_size < window_size")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            H, W = input_resolution

            img_mask = torch.zeros((1, H, W, 1))

            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask,
                self.window_size,
            )

            mask_windows = mask_windows.view(
                -1,
                self.window_size * self.window_size,
            )

            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            attn_mask = attn_mask.masked_fill(
                attn_mask != 0,
                float(-100.0),
            )
            attn_mask = attn_mask.masked_fill(
                attn_mask == 0,
                float(0.0),
            )
        else:
            attn_mask = None

        self.register_buffer(
            "attn_mask",
            attn_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape

        if L != H * W:
            raise ValueError(f"Input token length {L} does not match H*W={H*W}.")

        shortcut = x

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows = window_partition(
            shifted_x,
            self.window_size,
        )

        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C,
        )

        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask,
        )

        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C,
        )

        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W,
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Downsample HxW -> H/2 x W/2 and double channel dimension."""

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim

        self.reduction = nn.Linear(
            4 * dim,
            2 * dim,
            bias=False,
        )
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape

        if L != H * W:
            raise ValueError("Input feature has wrong token length.")

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"PatchMerging requires even resolution, got ({H}, {W}).")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat(
            [x0, x1, x2, x3],
            dim=-1,
        )

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """One hierarchical Swin Transformer stage."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | Sequence[float] = 0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        if isinstance(drop_path, Sequence):
            drop_path_values = list(drop_path)
        else:
            drop_path_values = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0 if i % 2 == 0 else window_size // 2),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_values[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchEmbed(nn.Module):
    """Image to non-overlapping patch tokens."""

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        patches_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        if (H, W) != self.img_size:
            raise ValueError(f"Expected input size {self.img_size}, got ({H}, {W}).")

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        return x


class SwinTransformerBackbone(nn.Module):
    """
    Hierarchical Swin Transformer encoder.

    Default parameters = Swin-T ImageNet-1K configuration.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer=nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = window_size

        self.num_layers = len(depths)
        self.ape = ape
        self.patch_norm = patch_norm

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.out_channels = tuple(int(embed_dim * 2**i) for i in range(self.num_layers))

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=(norm_layer if patch_norm else None),
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    num_patches,
                    embed_dim,
                )
            )
            trunc_normal_(
                self.absolute_pos_embed,
                std=0.02,
            )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=(PatchMerging if i_layer < self.num_layers - 1 else None),
                use_checkpoint=use_checkpoint,
            )

            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(
                module.weight,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.constant_(
                    module.bias,
                    0,
                )

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(
                module.bias,
                0,
            )
            nn.init.constant_(
                module.weight,
                1.0,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            bottleneck:
                [B, H/32 * W/32, 768] for Swin-T @ 224.

            skips:
                Token features before each encoder stage:
                stage 0: [B, 56*56, 96]
                stage 1: [B, 28*28, 192]
                stage 2: [B, 14*14, 384]
                stage 3: [B,  7*7,  768]
        """
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        skips: List[torch.Tensor] = []

        for layer in self.layers:
            skips.append(x)
            x = layer(x)

        x = self.norm(x)

        return x, skips

    def load_pretrained(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ):
        """
        Load an official Microsoft Swin Transformer checkpoint.

        Expected checkpoint:
            swin_tiny_patch4_window7_224.pth

        Classification-only parameters such as `head.*` are ignored.
        The encoder keys keep the same names as Microsoft's implementation.
        """
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        state_dict = checkpoint_data.get(
            "model",
            checkpoint_data,
        )

        # Remove common wrappers if present.
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]

            if key.startswith("backbone."):
                key = key[len("backbone.") :]

            # Classification head is not part of the segmentation backbone.
            if key.startswith("head."):
                continue

            cleaned_state_dict[key] = value

        model_state = self.state_dict()

        compatible_state = {}
        skipped = []

        for key, value in cleaned_state_dict.items():
            if key not in model_state:
                continue

            if model_state[key].shape != value.shape:
                skipped.append(
                    (
                        key,
                        tuple(value.shape),
                        tuple(model_state[key].shape),
                    )
                )
                continue

            compatible_state[key] = value

        msg = self.load_state_dict(
            compatible_state,
            strict=strict,
        )

        print(f"[Swin backbone] Loaded " f"{len(compatible_state)} tensors.")

        if skipped:
            print(
                f"[Swin backbone] Skipped " f"{len(skipped)} shape-mismatched tensors."
            )

        print(f"[Swin backbone] Missing keys: " f"{len(msg.missing_keys)}")
        print(f"[Swin backbone] Unexpected keys: " f"{len(msg.unexpected_keys)}")

        return msg
