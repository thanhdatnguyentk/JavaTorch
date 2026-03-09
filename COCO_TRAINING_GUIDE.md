# YOLO Training on COCO Dataset - Quick Start Guide

## Tổng quan

Hệ thống đã được cập nhật với tính năng tự động tải xuống COCO dataset và training YOLO.

## Các tính năng mới

### 1. **COCODatasetDownloader** 
- Tự động download COCO validation set (~1GB, 5000 ảnh)
- Có progress bar hiển thị tiến trình download
- Tự động extract files
- Kiểm tra xem dataset đã tồn tại chưa (không download lại nếu đã có)

### 2. **TrainYOLOCoco với Auto-Download**
- Tự động kiểm tra và download dataset nếu chưa có
- Sử dụng validation set thay vì train set (nhanh hơn cho testing)
- Tích hợp đầy đủ với progress bars và visualization

## Cách sử dụng

### Option 1: Download riêng, sau đó train

```bash
# Bước 1: Download COCO dataset (mất 10-30 phút tùy tốc độ mạng)
./gradlew :examples:downloadCOCO

# Bước 2: Train YOLO sau khi download xong
./gradlew :examples:trainYOLO
```

### Option 2: Chạy trực tiếp (tự động download nếu cần)

```bash
# Sẽ tự động download dataset nếu chưa có, sau đó bắt đầu training
./gradlew :examples:run
```

### Option 3: Custom parameters

```bash
# Cú pháp: <images_dir> <annotations_json> <epochs> <batch_size> <max_samples>
./gradlew :examples:run --args="data/coco/val2017 data/coco/annotations/instances_val2017.json 5 8 500"
```

## Chi tiết Dataset

- **Validation Set**: 5,000 ảnh, ~1GB (mặc định)
- **Train Set**: 118,000 ảnh, ~18GB (có thể dùng nếu cần)
- **Download từ**: http://cocodataset.org/

## Output Files

Sau khi training, các file sau sẽ được tạo:

- `coco_yolo_training_history.csv` - Training metrics
- `coco_yolo_training_curves.png` - All metrics visualization
- `coco_yolo_loss.png` - Loss curve only

## Cấu trúc thư mục sau khi download

```
data/
└── coco/
    ├── val2017/           # 5000 validation images
    │   ├── 000000000139.jpg
    │   ├── 000000000285.jpg
    │   └── ...
    └── annotations/
        ├── instances_val2017.json
        ├── instances_train2017.json (nếu download train set)
        └── ...
```

## Troubleshooting

### Download bị gián đoạn
- Xóa file `.zip` trong `data/coco/` và chạy lại
- Downloader sẽ tự động resume nếu folder đã có một phần files

### Không đủ dung lượng đĩa
- Validation set cần ~1.5GB (bao gồm zip và extracted)
- Train set cần ~25GB

### Muốn dùng dataset có sẵn
- Copy dataset vào đúng structure như trên
- Script sẽ tự động detect và không download lại

## Performance Tips

- **Batch size**: Tăng nếu có đủ RAM (default: 4)
- **Max samples**: Giảm để test nhanh (default: 200)
- **Epochs**: Tăng để train tốt hơn (default: 2)

## Example Commands

```bash
# Quick test với 50 samples, 1 epoch
./gradlew :examples:run --args="data/coco/val2017 data/coco/annotations/instances_val2017.json 1 4 50"

# Full validation set training
./gradlew :examples:run --args="data/coco/val2017 data/coco/annotations/instances_val2017.json 10 8 5000"

# Chỉ download dataset, không train
./gradlew :examples:downloadCOCO
```

## Notes

- Download lần đầu có thể mất 10-30 phút tùy tốc độ Internet
- Training với 200 samples mất khoảng 2-5 phút per epoch
- Kết quả training được lưu tự động ở thư mục gốc project
