# Mixer-SNN vs DMP-SNN: Digits và Fashion-MNIST dạng chuỗi (fmnist_stream)

Tài liệu so sánh hai bộ dữ liệu: **Digits** (kết quả cũ, điểm cao) và **fmnist_stream** (kết quả mới, điểm thấp hơn), để thấy bộ nào thật sự đo được khả năng nhớ quá khứ.

## 1. Kết quả

| | Digits | fmnist_stream |
|---|---|---|
| Mixer-SNN | 96,94% (bản `small`, 24,5 triệu tham số, 3 epoch) | 61,0% (bản `micro`, 0,66 triệu tham số, 8 epoch) |
| DMP-SNN | 91,67% (`d=100`, 83 nghìn tham số, 20 epoch) | 72,7% (`d=10`, 24 nghìn tham số, 15 epoch) |
| Chênh lệch (DMP − Mixer) | −5,3 điểm | **+11,7 điểm** |
| DMP khi tắt bộ nhớ chậm | không đo | 29,8% (từ 72,7%) |

Số liệu mới chỉ có **1 seed**.

Để đọc được mức tụt của Mixer ở fmnist_stream, cần một mốc tham chiếu: cùng ảnh Fashion-MNIST (3000 ảnh train) nhưng đưa cả ảnh một lần cho 4 bước (fmnist tĩnh) cho kết quả Mixer 79,0% và DMP 74,6%.

| Fashion-MNIST | Ảnh tĩnh | Đưa từng hàng, nhớ 21 hàng đầu | Mức tụt |
|---|---:|---:|---:|
| Mixer-micro | 79,0% | 61,0% | −18,0 điểm |
| DMP | 74,6% | 72,7% | −1,9 điểm |

## 2. Hai bộ dữ liệu khác nhau thế nào

| | Digits | fmnist_stream |
|---|---|---|
| Dữ liệu | Chữ số viết tay 8×8, 1797 ảnh (1437 train, 360 test) | Quần áo 28×28, 2000 ảnh train, 10000 ảnh test |
| Độ khó nội tại | Dễ: lớp tách rõ | Khó hơn: áo thun, áo len, áo khoác, sơ mi giống nhau |
| Mixer nhận | Ảnh phóng 64×64, `T=1` (không có thời gian) | 28 frame, frame *t* chỉ chứa hàng *t* |
| DMP nhận | 64 bước, mỗi bước 1 pixel | 28 bước, mỗi bước 1 hàng |
| Đọc đáp án ở đâu | Ảnh tĩnh, không cần nhớ | Chỉ 7 hàng cuối, nên phải nhớ 21 hàng đầu |
| Cần nhớ quá khứ? | **Không** (Mixer thậm chí không có chiều thời gian) | **Có** |

## 3. Vì sao Digits cao, fmnist_stream thấp

**Digits cao vì bài dễ và không đòi trí nhớ.**
* Ảnh nhỏ, 10 lớp tách nhau rõ. Theo hiểu biết chung, các mô hình đơn giản như k-NN hay SVM cũng đạt khoảng 97–99% trên bộ này (đây không phải số đo của dự án).
* Mixer chạy `T=1`: nó chỉ làm phân loại ảnh thường, nên 96,94% không nói gì về khả năng nhớ.
* Giao thức Digits lạc quan hơn: chọn checkpoint bằng chính tập test, dùng đủ 1437 ảnh, Mixer cỡ `small` và DMP `d=100` với ngưỡng nơ-ron chỉnh tay.

**fmnist_stream thấp hơn vì hai lý do cộng lại, mình chưa tách riêng được phần của từng lý do:**
1. Bài khó hơn (nhiều lớp giống nhau) và giao thức chặt hơn: chọn checkpoint bằng val, test đánh giá một lần, chỉ 2000 ảnh train, Mixer chỉ cỡ `micro`.
2. Bài đòi nhớ: thông tin phân loại nằm ở 21 hàng đầu mà đáp án chỉ đọc ở 7 hàng cuối.

Chính vì có lý do 1, **không thể lấy 96,94% − 61,0% rồi nói Mixer "kém đi 36 điểm vì không nhớ"**. Phép so sánh sạch hơn là bảng mức tụt ở mục 1: cùng ảnh, cùng 3000 ảnh train, chỉ khác cách đưa vào.

## 4. Điều rút ra được

* Mixer tụt 18 điểm khi phải nhớ hàng cũ, DMP tụt 2 điểm. Trên cùng bộ ảnh, đây là dấu hiệu ủng hộ giả thuyết Mixer chỉ có điện thế màng LIF (`tau=2`) nên quên nhanh, còn DMP có bộ nhớ chậm.
* DMP dùng bộ nhớ chậm thật: tắt nó đi thì điểm từ 72,7% xuống 29,8%.
* Digits không phân biệt được hai cơ chế nhớ này vì bài không cần nhớ. Mixer còn hơn DMP ở đó (96,94% so với 91,67%).

## 5. Hạn chế

* 1 seed, nên chênh lệch 11,7 điểm cần chạy thêm seed trước khi coi là chắc.
* Mixer ở fmnist_stream là cỡ `micro`, không phải cấu hình của paper. Một bản lớn hơn có thể ít tụt hơn.
* Hai bộ dùng giao thức, số epoch và kích thước mô hình khác nhau, nên điểm tuyệt đối không so thẳng được. Chỉ mức tụt trên cùng Fashion-MNIST là so sánh tương đối sạch.
* Chưa thử bước gộp hai phương pháp.

## 6. Tái lập

```bash
# fmnist_stream
python -m bench.run --dataset fmnist_stream --model mixer --mixer-size micro --epochs 8 --max-train 2000 --max-val 300 --out ./bench_out --data-root ./data
python -m bench.run --dataset fmnist_stream --model dmp --epochs 15 --max-train 2000 --max-val 300 --out ./bench_out --data-root ./data
# fmnist tĩnh (mốc tham chiếu)
python -m bench.run --dataset fmnist --model mixer --mixer-size micro --epochs 15 --max-train 3000 --max-val 500 --out ./bench_out --data-root ./data
python -m bench.run --dataset fmnist --model dmp --epochs 15 --max-train 3000 --max-val 500 --out ./bench_out --data-root ./data
```

Digits: xem lệnh trong `DIGITS_COMPARISON.md`. Kết quả thô nằm trong `bench_out/results/*.json`.
