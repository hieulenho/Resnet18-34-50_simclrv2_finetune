# ResNet50 + SimCLRv2 on STL10

Dự án này nâng cấp từ baseline ResNet18 + SimCLR sang pipeline gần với SimCLRv2 hơn, sử dụng ResNet50 làm backbone chính, projection head 3 tầng MLP, contrastive pretraining, linear evaluation và fine-tuning.

Mục tiêu chính của project là nghiên cứu vai trò của data augmentation trong CNNs, đặc biệt trong contrastive learning. Trong SimCLR/SimCLRv2, data augmentation không chỉ đóng vai trò tăng dữ liệu, mà còn là cơ chế tạo hai view khác nhau của cùng một ảnh để hình thành positive pair cho contrastive loss.

## 1. Tính năng chính

* Hỗ trợ các backbone: `resnet18`, `resnet34`, `resnet50`.
* ResNet18/ResNet34 dùng `BasicBlock`.
* ResNet50 dùng `Bottleneck`.
* Hỗ trợ SimCLR/SimCLRv2-style pretraining.
* Projection head mặc định là 3-layer MLP.
* Hỗ trợ:

  * `pretrain`
  * `finetune`
  * `linear-eval`
  * `supervised`
* Fine-tune có thể chọn số layer của projection head được giữ lại:

  * `--finetune-proj-layers 0`: backbone -> classifier
  * `--finetune-proj-layers 1`: backbone -> projector layer 1 -> classifier
  * `--finetune-proj-layers 2`: backbone -> projector layer 1 + 2 -> classifier

Trong các thí nghiệm hiện tại trên STL10 full-label, cấu hình tốt nhất đang là:

```text
ResNet50 + SimCLRv2 pretrain 300 epochs
fine-tune with finetune_proj_layers = 0
lr = 0.02
best validation accuracy = 82.65%
```

## 2. Cấu trúc file

```text
train_resnet.py
ssl_simclr.py
dataset.py
requirements.txt
README.md
resnet.sh

resnet18/
  basicblock.py
  resnet18.py
  __init__.py

checkpoints/
  simclrv2_resnet50_epoch_300.pt

checkpoints_finetune_ft0_lr002/
  *.pt

paper_outputs/
  finetune_eval/
  linear_eval_memorysafe/
```

Các notebook phân tích kết quả:

```text
resnet_finetune_eval_existing_checkpoint.ipynb
resnet_lineareval_memorysafe.ipynb
```

## 3. Cài đặt

Trên local:

```bash
pip install -r requirements.txt
```

Trên HPC01, Python hệ thống nên gọi bằng full path:

```bash
/opt/apps/python/3.8.10/bin/python3
```

Nếu `python3` không nhận trên compute node, trong file `.sh` cần khai báo:

```bash
PYTHON=/opt/apps/python/3.8.10/bin/python3
```

và chạy bằng:

```bash
$PYTHON train_resnet.py ...
```

## 4. Dataset

Dataset chính:

```text
STL10
image_size = 96
num_classes = 10
```

Cấu trúc dữ liệu:

```text
data/
  stl10_binary/
    train_X.bin
    train_y.bin
    test_X.bin
    test_y.bin
    unlabeled_X.bin
```

Nếu dùng `--download`, code sẽ tự tải STL10 khi môi trường có internet.

## 5. Pretrain SimCLRv2

Cấu hình pretrain chính đã dùng trong thí nghiệm:

```bash
python train_resnet.py \
  --mode pretrain \
  --arch resnet50 \
  --dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --epochs 300 \
  --batch-size 128 \
  --optimizer lars \
  --lr 0.3 \
  --warmup-epochs 15 \
  --weight-decay 1e-4 \
  --proj-layers 3 \
  --proj-dim 128 \
  --temperature 0.2 \
  --num-workers 1 \
  --out-dir ./checkpoints \
  --save-every 20 \
  --log-every 20
```

Checkpoint thu được:

```text
checkpoints/simclrv2_resnet50_epoch_300.pt
```

Thông tin checkpoint:

```text
epoch = 300
mode = pretrain
arch = resnet50
dataset = stl10
batch_size = 128
optimizer = lars
lr = 0.3
temperature = 0.2
projection head = 3-layer MLP
```

## 6. Fine-tune từ checkpoint pretrain

Cấu hình fine-tune tốt nhất hiện tại:

```bash
python train_resnet.py \
  --mode finetune \
  --arch resnet50 \
  --dataset stl10 \
  --labeled-dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --num-classes 10 \
  --ckpt ./checkpoints/simclrv2_resnet50_epoch_300.pt \
  --epochs 120 \
  --batch-size 64 \
  --optimizer sgd \
  --lr 0.02 \
  --warmup-epochs 0 \
  --weight-decay 0 \
  --proj-layers 3 \
  --finetune-proj-layers 0 \
  --num-workers 2 \
  --out-dir ./checkpoints_finetune_ft0_lr002 \
  --save-every 10 \
  --log-every 20
```

Ý nghĩa:

```text
--finetune-proj-layers 0
```

tức là classifier được gắn trực tiếp sau backbone:

```text
ResNet50 backbone -> classifier
```

Kết quả hiện tại:

```text
best_acc = 82.65%
epoch = 120
lr = 0.02
finetune_proj_layers = 0
```

## 7. Fine-tune từ lớp giữa projection head

Để kiểm tra giả thuyết của SimCLRv2 về việc sử dụng middle layer của projection head trong downstream task, có thể chạy:

```bash
python train_resnet.py \
  --mode finetune \
  --arch resnet50 \
  --dataset stl10 \
  --labeled-dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --num-classes 10 \
  --ckpt ./checkpoints/simclrv2_resnet50_epoch_300.pt \
  --epochs 120 \
  --batch-size 64 \
  --optimizer sgd \
  --lr 0.02 \
  --warmup-epochs 0 \
  --weight-decay 0 \
  --proj-layers 3 \
  --finetune-proj-layers 1 \
  --num-workers 2 \
  --out-dir ./checkpoints_finetune_ft1_lr002 \
  --save-every 10 \
  --log-every 20
```

Khi đó pipeline là:

```text
ResNet50 backbone -> projector layer 1 -> classifier
```

Có thể thử thêm:

```bash
--finetune-proj-layers 2
```

để kiểm tra việc sử dụng sâu hơn trong projection head.

## 8. Linear Evaluation

Linear evaluation dùng để đánh giá chất lượng representation học được sau pretrain. Trong chế độ này, backbone được freeze và chỉ train classifier tuyến tính.

Notebook khuyên dùng:

```text
resnet_lineareval_memorysafe.ipynb
```

Notebook này chạy theo cách tiết kiệm bộ nhớ:

```text
1. Load checkpoint pretrain.
2. Freeze backbone.
3. Extract feature bằng torch.no_grad().
4. Cache feature ra file.
5. Train linear classifier trên feature đã cache.
6. Vẽ biểu đồ loss, accuracy, confusion matrix, per-class accuracy.
```

Output:

```text
paper_outputs/linear_eval_memorysafe/
  figures/
  tables/
  checkpoints/
```

## 9. Phân tích fine-tune checkpoint

Notebook khuyên dùng:

```text
resnet_finetune_eval_existing_checkpoint.ipynb
```

Notebook này không train lại. Nó chỉ load checkpoint fine-tune đã có và xuất:

```text
validation top-1 accuracy
validation top-5 accuracy
confusion matrix
normalized confusion matrix
per-class accuracy
confidence histogram
error rate by class
summary CSV
```

Output:

```text
paper_outputs/finetune_eval/
  figures/
  tables/
```

## 10. Chạy trên HPC01 bằng PBS

Thư mục project trên server:

```bash
/datausers/cic/tqviet/simclrv2_resnet
```

Queue ổn định hiện tại:

```text
long_gpu
```

Header PBS khuyên dùng:

```bash
#!/bin/bash
#PBS -N simclrv2_resnet50_pretrain
#PBS -j oe
#PBS -m abe
#PBS -M hieulenho8@gmail.com
#PBS -q long_gpu
#PBS -l select=1:ncpus=3:ngpus=1:mem=30G
```

Nên dùng Python full path:

```bash
PYTHON=/opt/apps/python/3.8.10/bin/python3
```

Ví dụ chạy pretrain trên HPC:

```bash
$PYTHON train_resnet.py \
  --mode pretrain \
  --arch resnet50 \
  --dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --epochs 300 \
  --batch-size 128 \
  --optimizer lars \
  --lr 0.3 \
  --warmup-epochs 15 \
  --weight-decay 1e-4 \
  --proj-layers 3 \
  --proj-dim 128 \
  --temperature 0.2 \
  --num-workers 1 \
  --out-dir ./checkpoints \
  --save-every 20 \
  --log-every 20
```

Submit job:

```bash
qsub resnet.sh
qstat -u $USER
```

Theo dõi log:

```bash
tail -f $(ls -t logs/*.log | head -n 1)


## 11. Kết quả hiện tại

| Method               | Backbone |   Pretrain | Downstream mode | Projection layer |      Accuracy |
| -------------------- | -------- | ---------: | --------------- | ---------------: | ------------: |
| ResNet18 + SimCLR cũ | ResNet18 |     SimCLR | Linear eval     |                - |        78.50% |
| ResNet50 + SimCLRv2  | ResNet50 | 300 epochs | Fine-tune       |                0 |        82.65% |
| ResNet50 + SimCLRv2  | ResNet50 | 300 epochs | Fine-tune       |                1 | cần chạy thêm |
| ResNet50 + SimCLRv2  | ResNet50 | 300 epochs | Linear eval     |  frozen backbone |  cần cập nhật |

## 12. Các thực nghiệm nên chạy thêm

Các thực nghiệm ưu tiên:



Ví dụ weight decay ablation:

```bash
python train_resnet.py \
  --mode finetune \
  --arch resnet50 \
  --dataset stl10 \
  --labeled-dataset stl10 \
  --data-root ./data \
  --image-size 96 \
  --num-classes 10 \
  --ckpt ./checkpoints/simclrv2_resnet50_epoch_300.pt \
  --epochs 120 \
  --batch-size 64 \
  --optimizer sgd \
  --lr 0.02 \
  --warmup-epochs 0 \
  --weight-decay 1e-4 \
  --proj-layers 3 \
  --finetune-proj-layers 0 \
  --num-workers 2 \
  --out-dir ./checkpoints_finetune_ft0_lr002_wd1e4 \
  --save-every 10 \
  --log-every 20
```

## 13. Ghi chú quan trọng

* `--batch-size` là batch size trên một GPU.
* Với SimCLR/SimCLRv2, batch lớn làm tăng số negative samples nhưng cũng tăng nhu cầu VRAM.
* Với Tesla P100 16GB, ResNet50 pretrain nên bắt đầu với `batch-size=128`.
* Với fine-tune local trên RTX 4060 8GB, nên dùng `batch-size=64`; nếu OOM thì hạ xuống 32.
* Pretrain loss không thể so sánh tuyệt đối giữa các batch size khác nhau vì số negative samples thay đổi.
* Checkpoint pretrain không có classifier, nên không dùng trực tiếp để tính validation accuracy classification.
* Accuracy classification phải lấy từ fine-tune checkpoint hoặc linear evaluation.
