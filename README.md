# poply-image-segmentation

U-Net (ResNet18/34/50/101) và Swin-Unet V2 cho phân đoạn 3 lớp:
background, non-neoplastic, neoplastic.

## Chuẩn bị dữ liệu

```powershell
python -m pip install -r requirements.txt
python scripts/dataloaders.py
# Hoặc: python -m scripts.dataloaders
```

Mặc định tải `avcuongy/bkai-igh-neopolyp-small-sample` từ Kaggle vào
`data/kagglehub/data/`, gồm `train/`, `gt/`, `sample/{train,valid,test}.csv`.
Log nằm ở `data/log/dataloaders.log`, tính từ thư mục dự án dù chạy ở thư mục khác.
Script kiểm tra mọi đường dẫn ảnh/mask trong cả ba CSV; nếu đầy đủ thì không tải lại.
Dùng `--force` để tải lại, `--output-dir` để đổi nơi lưu. Lỗi tải/CSV/file thiếu
được ghi kèm traceback vào log, tiến trình trả mã lỗi 1.
Không cần tải trọng số pretrained để chuẩn bị dữ liệu.

## Các module trong src

| Phần | Chức năng |
| --- | --- |
| `model.unet`, `model.swin_unet` | Backbone, decoder, logits và feature map |
| `model.contrastive` | `ContrastiveUNet`, `ContrastiveSwinUNet`, projection head |
| `method` | `MeanTeacher`, `MeanTeacherInfoNCE`, vòng lặp train supervised/MT |
| `loss` | GDL, weighted CE + GDL, softmax MSE, pixel/region contrastive loss |
| `data` | Tải dữ liệu, giải mã mask, augmentation, DataLoader |
| `evaluation` | Đọc checkpoint, confusion matrix và các chỉ số |

Ba phương pháp supervised, Mean Teacher và Mean Teacher + contrastive đều dùng
được với U-Net/Swin-Unet. Nhánh contrastive giữ thuật toán trong notebook:
memory pixel/region, lấy mẫu hard/random, semi-hard mining, warmup; chỉ dùng ảnh
có nhãn. Teacher chỉ EMA các tham số segmentation; BatchNorm của Teacher cập
nhật từ weak view như notebook. Projection head không được đưa vào EMA.

Để import các module trực tiếp trong Python/notebook, thêm thư mục `src` vào
`sys.path`, hoặc đặt `$env:PYTHONPATH = "$PWD/src"` trong PowerShell.

```python
from data.loaders import build_dataloaders
from loss import WeightedCE_GDL_Loss
from method import MeanTeacher, train_mean_teacher_one_epoch
from model.unet import UNet
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
loaders = build_dataloaders(method="mean_teacher", subset="half")
method = MeanTeacher(UNet(pretrained=False)).to(device)
criterion = WeightedCE_GDL_Loss([0.5, 2.5, 1.0]).to(device)
optimizer = torch.optim.AdamW(method.student.parameters(), lr=3e-4)
metrics, global_step = train_mean_teacher_one_epoch(
    method, loaders["labeled"], loaders["unlabeled"], optimizer, criterion,
    epoch=1, mt_config={"max_consistency_weight": 4, "consistency_rampup_epochs": 10},
    device=device,
)
```

Với contrastive: tạo `ContrastiveUNet` hoặc `ContrastiveSwinUNet` bằng bản sao
`loss.DEFAULT_CONTRASTIVE_CONFIG`, rồi bọc bằng `MeanTeacherInfoNCE(student,
image_keys)`. `image_keys` phải chứa định danh ảnh có nhãn duy nhất, đúng thứ tự
`loaders['labeled'].dataset.df`. Dùng cùng vòng lặp MT ở trên. Lưu
`method.state_dict()` để giữ cả memory, đồng thời lưu optimizer, epoch,
global_step và cấu hình khi cần tiếp tục huấn luyện. Swin cần giữ cấu hình
kiến trúc trong checkpoint. Các vòng lặp này chạy FP32; scheduler/AMP do caller quản lý.

## Evaluation

```powershell
python scripts/evaluate.py "models/baseline/best_UNet_half.pth" --model-type unet
```

Chọn `--model-type` trong `unet`, `swin`, `mean_teacher_unet`, `mean_teacher_swin`.
Checkpoint contrastive dùng cùng model-type theo backbone; projection head
được bỏ qua khi inference, còn các trọng số segmentation phải khớp hoàn toàn.
MT mặc định dùng teacher; `--branch student` để đánh giá student.
Kết quả lưu ở `output/evaluation/metrics.csv` và `confusion_matrix.csv`.

API `evaluation.evaluate_model` trả bảng theo lớp, `macro_fg`, `macro_all` và
confusion matrix (hàng = nhãn thật, cột = dự đoán). Dice/F1, IoU, precision,
recall/sensitivity, specificity và accuracy được tính từ tổng pixel cả tập.
`accuracy` là one-vs-rest theo lớp; `pixel_accuracy` là tổng dự đoán đúng / tổng
pixel hợp lệ. Tỉ lệ không xác định trả NaN và không tham gia macro average.
HD95 lấy max của percentile 95 hai hướng, trung bình theo ảnh, đơn vị pixel:
cả hai mask rỗng → NaN, một mask rỗng → phạt bằng đường chéo ảnh. `ignore_index`
mặc định -100 bị loại khỏi confusion và mask tính HD95.

## Kiểm tra

```powershell
python -m unittest discover -s qa -v
```

Kiểm tra trên CPU: giá trị metric tính tay, HD95, loss/gradient, toàn bộ nhãn
bị bỏ qua, memory contrastive, optimizer/EMA, lưu-khôi phục memory, và luồng tải
mới/chạy lại/thiếu file/mất mạng bằng dữ liệu giả lập. Không tải pretrained weights.
