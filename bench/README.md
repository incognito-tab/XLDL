# Benchmark: Mixer-SNN vs DMP-SNN — khả năng "nhớ quá khứ"

Mục tiêu: so sánh hai cách một SNN giữ thông tin theo thời gian, rồi tìm điểm yếu của từng bài báo theo từng bộ dữ liệu.

| | Mixer-SNN (arXiv 2306.12465) | DMP-SNN (arXiv 2512.07602) |
|---|---|---|
| Cơ chế nhớ | Chỉ có điện thế màng LIF (`tau = 2`, quên nhanh). Ảnh tĩnh được lặp lại `T` lần | LIF nhanh + **bộ nhớ chậm LMU** `m[k] = Ā·m[k-1] + B̄·u[k]` với `d` chiều và cửa sổ `theta` bước |
| Thế mạnh | Trích xuất đặc trưng không gian (patch, token/channel mixing) | Chuỗi dài, ít tham số |
| Nghi vấn | Không có cơ chế nhớ dài hạn | Bộ nhớ chỉ được nạp bởi **1 số vô hướng** `u = ReLU(W_u·x)` mỗi lớp; cửa sổ `theta` cố định |

Mã gốc của hai bài báo **không bị sửa**: `bench/nets.py` gọi trực tiếp `models/mixers_sparse_patchcell_origin.sMLPNet` (mô hình mà `train.py` dùng) và `SpikingLMUFFTCell` trong `Dual_memory_pathways/smnist/src/dual_pathway.py`.

## Bộ dữ liệu

| Tên | Dữ liệu | Mixer nhận | DMP nhận | Đọc kết quả | Câu hỏi |
|---|---|---|---|---|---|
| `fmnist` | Fashion-MNIST | Ảnh 32×32, lặp `T=4` | 28 bước, mỗi bước 1 hàng | trung bình mọi bước | Mức trần khi không cần nhớ |
| `fmnist_stream` | Fashion-MNIST | 28 frame, frame *t* chỉ có hàng *t* | như trên | **7 bước cuối** | Khi phải nhớ 21 hàng đầu thì còn bao nhiêu? |
| `cifar10` | CIFAR-10 (crop + flip) | Ảnh 32×32, `T=4` | 32 bước × 96 (1 hàng RGB) | trung bình | Dữ liệu không gian khó → nút thắt của DMP |
| `shd` | Spiking Heidelberg Digits, 20 lớp | 100 frame (140 kênh trải lên canvas 32×32) | 100 bước × 140 kênh (như repo DMP) | trung bình | Dữ liệu thời gian thật |
| `recall` | Tổng hợp: cue ở đầu → trễ D bước (nhiễu + 2 cue gây nhiễu) → hỏi | 32×32 canvas mỗi bước | 20 kênh mỗi bước | 5 bước hỏi | Accuracy theo độ trễ D |

Tiền xử lý SHD giống repo DMP: 100 bin trong 1 giây, max-pool 700 → 140 kênh, spike nhị phân.

## Giao thức (giống nhau cho cả hai mô hình)

* Tập train chính thức → 90% train / 10% validation (chia phân tầng, cố định). Tập test chỉ đánh giá **một lần**, bằng checkpoint có val tốt nhất.
* AdamW + cosine (OneCycle). Mixer: lr 1e-3, wd 0.01, AMP. DMP: lr 5e-3, wd 0.1 (theo script SHD của DMP).
* Mixer `small` = cấu hình của bài báo (dim 78, depths [2, 8, 14, 2]). Mixer `lite` = cùng các khối, dim 48, depths [1, 2, 2, 1], dùng cho chuỗi dài để vừa thời gian GPU.
* DMP: 2 lớp × 128 nơ-ron, `d = 10`, `theta = 40` cho SHD, `theta =` độ dài chuỗi cho các bộ khác.
* Mỗi kết quả ghi: test acc, val acc, số tham số, thời gian/epoch, firing rate (độ thưa xung). Với DMP còn có `test_acc_memory_off`: cùng trọng số đã train nhưng tắt bộ nhớ chậm lúc test.

## Chạy

```bash
python -m bench.run --dataset shd --model dmp --seed 0 --out ./bench_out
python -m bench.run --dataset cifar10 --model mixer --mixer-size small --T 4 --out ./bench_out
python -m bench.summarize --out ./bench_out        # summary.md, summary.csv, plots/
```

Trên Colab: mở `bench/colab_run.ipynb` (đã chia sẵn các khối A–D theo thứ tự ưu tiên). Cần `spikingjelly==0.0.0.0.14`; các bản mới hơn đã bỏ `spikingjelly.clock_driven` mà mã DMP cần.
