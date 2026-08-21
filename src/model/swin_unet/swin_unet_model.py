"""
Swin-Unet model using a Swin Transformer encoder backbone.

Architecture adapted from the official Swin-Unet implementation:
https://github.com/HuCaoFighting/Swin-Unet

Encoder / pretrained weights:
https://github.com/microsoft/Swin-Transformer

The default model is Swin-Unet with a Swin-T encoder:
    img_size = 224
    patch_size = 4
    embed_dim = 96
    depths = (2, 2, 6, 2)
    num_heads = (3, 6, 12, 24)
    window_size = 7
"""

from __future__ import annotations

import copy
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .swin_transformer import (
    SwinTransformerBackbone,
    SwinTransformerBlock,
)


class PatchExpand(nn.Module):
    """2x token-space upsampling used by the Swin-Unet decoder."""

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        dim_scale: int = 2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        self.expand = (
            nn.Linear(
                dim,
                2 * dim,
                bias=False,
            )
            if dim_scale == 2
            else nn.Identity()
        )

        self.norm = norm_layer(dim // dim_scale)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        H, W = self.input_resolution

        x = self.expand(x)

        B, L, C = x.shape

        if L != H * W:
            raise ValueError(f"Input token length {L} != H*W={H*W}.")

        x = x.view(B, H, W, C)

        # Equivalent to pixel shuffle in token/channel layout.
        x = x.view(
            B,
            H,
            W,
            2,
            2,
            C // 4,
        )

        x = (
            x.permute(
                0,
                1,
                3,
                2,
                4,
                5,
            )
            .contiguous()
            .view(
                B,
                H * 2,
                W * 2,
                C // 4,
            )
        )

        x = x.view(
            B,
            -1,
            C // 4,
        )

        x = self.norm(x)

        return x


class FinalPatchExpandX4(nn.Module):
    """Final 4x expansion from patch resolution back to image resolution."""

    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        dim_scale: int = 4,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale

        self.expand = nn.Linear(
            dim,
            (dim_scale**2) * dim,
            bias=False,
        )

        self.norm = norm_layer(dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        H, W = self.input_resolution

        x = self.expand(x)

        B, L, C = x.shape

        if L != H * W:
            raise ValueError(f"Input token length {L} != H*W={H*W}.")

        scale = self.dim_scale

        x = x.view(
            B,
            H,
            W,
            scale,
            scale,
            C // (scale**2),
        )

        x = (
            x.permute(
                0,
                1,
                3,
                2,
                4,
                5,
            )
            .contiguous()
            .view(
                B,
                H * scale,
                W * scale,
                C // (scale**2),
            )
        )

        x = x.view(
            B,
            -1,
            self.dim,
        )

        x = self.norm(x)

        return x


class BasicLayerUp(nn.Module):
    """One Swin Transformer decoder stage followed optionally by PatchExpand."""

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
        upsample=None,
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

        if upsample is not None:
            self.upsample = upsample(
                input_resolution=input_resolution,
                dim=dim,
                dim_scale=2,
                norm_layer=norm_layer,
            )
        else:
            self.upsample = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class SwinUNet(nn.Module):
    """
    Swin-Unet with a configurable Swin encoder.

    Defaults match Swin-T + 3-class semantic segmentation.

    Args:
        img_size:
            Fixed model input size. Official Swin-Unet commonly uses 224.

        num_classes:
            Segmentation classes. Use 3 for:
            background, non-neoplastic, neoplastic.

        pretrained_path:
            Optional Microsoft Swin-T ImageNet-1K checkpoint path.

        pretrained_decoder:
            If True, encoder stage weights are mirrored into compatible
            decoder Swin blocks following the initialization idea used by
            the official Swin-Unet implementation.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 3,
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
        pretrained_path: str | None = None,
        pretrained_decoder: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = window_size
        self.num_layers = len(depths)

        # ---------------------------------------------------------
        # Encoder: Microsoft Swin Transformer-compatible backbone
        # ---------------------------------------------------------
        self.encoder = SwinTransformerBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
        )

        patches_resolution = self.encoder.patches_resolution

        # Same stochastic-depth schedule as encoder.
        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()

        for i_layer in range(self.num_layers):
            encoder_stage = self.num_layers - 1 - i_layer

            dim = int(embed_dim * 2**encoder_stage)

            resolution = (
                patches_resolution[0] // (2**encoder_stage),
                patches_resolution[1] // (2**encoder_stage),
            )

            if i_layer == 0:
                self.concat_back_dim.append(nn.Identity())

                self.layers_up.append(
                    PatchExpand(
                        input_resolution=resolution,
                        dim=dim,
                        dim_scale=2,
                        norm_layer=norm_layer,
                    )
                )

            else:
                self.concat_back_dim.append(
                    nn.Linear(
                        2 * dim,
                        dim,
                    )
                )

                stage_dpr = dpr[
                    sum(depths[:encoder_stage]) : sum(depths[: encoder_stage + 1])
                ]

                self.layers_up.append(
                    BasicLayerUp(
                        dim=dim,
                        input_resolution=resolution,
                        depth=depths[encoder_stage],
                        num_heads=num_heads[encoder_stage],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=stage_dpr,
                        norm_layer=norm_layer,
                        upsample=(
                            PatchExpand if i_layer < self.num_layers - 1 else None
                        ),
                        use_checkpoint=use_checkpoint,
                    )
                )

        self.norm_up = norm_layer(embed_dim)

        self.final_up = FinalPatchExpandX4(
            input_resolution=patches_resolution,
            dim=embed_dim,
            dim_scale=4,
            norm_layer=norm_layer,
        )

        self.segmentation_head = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=num_classes,
            kernel_size=1,
            bias=False,
        )

        self.apply(self._init_decoder_weights)

        if pretrained_path is not None:
            self.load_pretrained(
                pretrained_path,
                load_decoder=pretrained_decoder,
            )

    @staticmethod
    def _init_decoder_weights(
        module: nn.Module,
    ) -> None:
        # Encoder initializes itself. Reinitializing it with the same rule
        # is harmless before loading pretrained weights.
        try:
            from timm.layers import trunc_normal_
        except ImportError:
            from timm.models.layers import trunc_normal_

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

    def forward_up_features(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
                continue

            skip_index = self.num_layers - 1 - i

            skip = skips[skip_index]

            if x.shape[1] != skip.shape[1]:
                raise ValueError(
                    f"Decoder/skip token mismatch at stage {i}: "
                    f"{x.shape} vs {skip.shape}"
                )

            x = torch.cat(
                [x, skip],
                dim=-1,
            )

            x = self.concat_back_dim[i](x)
            x = layer_up(x)

        x = self.norm_up(x)

        return x

    def tokens_to_feature_map(
        self,
        x: torch.Tensor,
        resolution: Tuple[int, int],
    ) -> torch.Tensor:
        B, L, C = x.shape
        H, W = resolution

        if L != H * W:
            raise ValueError(f"Token length {L} != H*W={H*W}.")

        return x.transpose(1, 2).contiguous().view(B, C, H, W)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, " f"got {x.shape[1]}."
            )

        bottleneck, skips = self.encoder(x)

        decoder_tokens = self.forward_up_features(
            bottleneck,
            skips,
        )

        x = self.final_up(decoder_tokens)

        B, L, C = x.shape

        if L != self.img_size * self.img_size:
            raise ValueError(
                "Final token count does not match output image resolution."
            )

        x = x.view(
            B,
            self.img_size,
            self.img_size,
            C,
        )

        x = x.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        logits = self.segmentation_head(x)

        if not return_features:
            return logits

        bottleneck_resolution = (
            self.encoder.patches_resolution[0] // (2 ** (self.num_layers - 1)),
            self.encoder.patches_resolution[1] // (2 ** (self.num_layers - 1)),
        )

        bottleneck_map = self.tokens_to_feature_map(
            bottleneck,
            bottleneck_resolution,
        )

        decoder_map = self.tokens_to_feature_map(
            decoder_tokens,
            self.encoder.patches_resolution,
        )

        return logits, {
            "bottleneck": bottleneck_map,
            "decoder": decoder_map,
        }

    def load_pretrained(
        self,
        checkpoint_path: str,
        load_decoder: bool = True,
    ):
        """
        Load Microsoft Swin checkpoint into encoder.

        If load_decoder=True, compatible encoder Swin block weights are
        mirrored into decoder stages, following the official Swin-Unet
        pretraining strategy.
        """

        # 1) Load exact Microsoft-compatible keys into the encoder.
        encoder_msg = self.encoder.load_pretrained(
            checkpoint_path,
            strict=False,
        )

        if not load_decoder:
            return encoder_msg

        # 2) Read checkpoint again to initialize compatible decoder blocks.
        checkpoint_data = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        state_dict = checkpoint_data.get(
            "model",
            checkpoint_data,
        )

        cleaned = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]

            if key.startswith("backbone."):
                key = key[len("backbone.") :]

            if key.startswith("head."):
                continue

            cleaned[key] = value

        model_state = self.state_dict()

        decoder_state = {}

        # Map encoder layer N -> decoder layer (3-N).
        # Only keys with exact matching names/shapes are loaded.
        for key, value in cleaned.items():
            if not key.startswith("layers."):
                continue

            parts = key.split(".")

            if len(parts) < 3:
                continue

            encoder_stage = int(parts[1])
            decoder_stage = self.num_layers - 1 - encoder_stage

            new_key = f"layers_up.{decoder_stage}." + ".".join(parts[2:])

            if new_key not in model_state:
                continue

            if model_state[new_key].shape != value.shape:
                continue

            decoder_state[new_key] = value

        decoder_msg = self.load_state_dict(
            decoder_state,
            strict=False,
        )

        print(f"[Swin-Unet decoder] Loaded " f"{len(decoder_state)} mirrored tensors.")

        return {
            "encoder": encoder_msg,
            "decoder": decoder_msg,
        }
