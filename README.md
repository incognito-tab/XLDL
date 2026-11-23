# Nhớ video theo thời gian thực bằng mạng nơ-ron xung

Dự án xây một mạng nơ-ron xung (SNN) xem video thật đang chạy và trả lời câu hỏi về những gì đã qua. Mô hình không được xem lại khung hình cũ, nên phải tự nhớ. Mô hình nâng cấp của nhóm được so sánh công bằng với Mixer-SNN và DMP-SNN.

Kế hoạch ban đầu nằm ở [docs/plan.html](docs/plan.html). Tải file về rồi mở bằng trình duyệt.

## Kết quả mới nhất

Cả ba mô hình dùng cùng bộ mã hóa ảnh (Spiking ResNet18), cùng dữ liệu, cùng 60 epoch và cùng đầu trả lời câu hỏi. Mô hình được chọn theo tập validation, rồi chấm một lần trên 2.048 luồng test.

| Mô hình | Tham số | Validation | Test |
|---|---:|---:|---:|
| **Bản nâng cấp** | 0,83 triệu | **63,7%** | **61,6%** |
| Mixer-SNN tốt nhất (`tau = 24`, có đường đọc, 448 nơ-ron) | 0,82 triệu | 37,4% | 34,5% |
| DMP-SNN tốt nhất (`d = 40`, `theta = 256`) | 0,34 triệu | 31,7% | 28,4% |

Mức đoán bừa trung bình là 9,4%. Một bộ nhớ hoàn hảo dùng bộ mã hóa này cũng chỉ đạt khoảng 74,7%, đó là mức trần.

Theo từng loại câu hỏi trên test:

| Loại câu hỏi | Mixer | DMP | Nâng cấp |
|---|---:|---:|---:|
| Cách đây N giây đang có hoạt động gì? | 17,6% | 20,5% | 65,6% |
| Ở giây thứ T đang xảy ra hoạt động gì? | 28,1% | 24,2% | 69,0% |
| Trong 4 hoạt động này, hoạt động nào đã xuất hiện? | 61,2% | 49,8% | 73,3% |
| Ngay sau lần gần nhất có X là hoạt động gì? | 31,3% | 19,2% | 38,6% |

Theo độ xa của sự kiện được hỏi:

| Sự kiện cách đây | Mixer | DMP | Nâng cấp |
|---|---:|---:|---:|
| 0 đến 4 giây | 58,3% | 34,8% | 62,4% |
| 16 đến 32 giây | 7,8% | 20,0% | 61,5% |

Tóm tắt:

- **So với Mixer và DMP:** bản nâng cấp đúng gấp 1,79 lần Mixer và 2,17 lần DMP.
- **Nhớ xa:** độ chính xác gần như không giảm theo thời gian. Mixer quên gần hết sau 16 giây.
- **Bằng chứng dùng bộ nhớ:** tắt bộ nhớ thì bản nâng cấp rơi xuống 9,2%, gần mức đoán bừa.
- **Giới hạn:** mỗi mô hình mới chạy 1 seed.

### Bản nâng cấp khác gì

Bản nâng cấp giữ bộ nhớ Legendre chậm của DMP và thêm bốn thay đổi:

| | Thay đổi | Lý do |
|---|---|---|
| H1 | Bộ nhớ nhận một vector 16 kênh thay vì 1 số | Giữ được nhiều thông tin về hoạt động hơn |
| H2 | Nhớ trên nhiều cửa sổ cùng lúc: 16, 64 và 256 bước | Vừa nhớ chi tiết gần, vừa nhớ xa |
| H3 | Câu hỏi đọc thẳng vào bộ nhớ qua một lớp nơ-ron xung | Mô hình biết cần lấy phần nào của quá khứ |
| H4 | Đọc bộ nhớ đúng thời điểm được hỏi bằng đa thức Legendre cố định | Giúp câu "cách đây N giây" và "ở giây thứ T" |

Câu hỏi được đưa vào dưới dạng xung, gồm đồng hồ hiện tại và khoảng cách từ thời điểm được hỏi tới hiện tại. Áp dụng cho cả ba mô hình. Mọi tín hiệu đi giữa các nơ-ron đều là xung, và có kiểm thử bảo đảm không mô hình nào nhìn trước tương lai.

### Các đợt huấn luyện

| Đợt | Thay đổi | Nâng cấp | Mixer | DMP | Thư mục |
|---|---|---:|---:|---:|---|
| 1 | Bộ mã hóa sMLPNet `lite`, chỉ chấm validation | | | | `results/memory/`, `results/final/` |
| 2 | Bộ mã hóa sMLPNet `small`, 3 seed | 43,8% | 28,9% | 23,9% | `results/round2/` |
| 3 | Spiking ResNet18, 60 epoch, có đồng hồ | 56,6% | 33,9% | 26,1% | `results/round3/` |
| 4 | Thêm câu hỏi dạng "cách đây bao lâu" và H4 | **61,6%** | **34,5%** | **28,4%** | `results/round4/` |

Mỗi file JSON trong `results/` là một lần chạy:

- `config`: cấu hình.
- `params`: số tham số.
- `history`: điểm validation theo epoch.
- `val`, `test`: điểm cuối, gồm:
  - `mean_acc`: trung bình 4 loại câu hỏi;
  - `above_chance`: phần vượt mức đoán bừa;
  - `by_kind`: theo loại câu hỏi;
  - `by_age`: theo độ xa;
  - `firing_rate`: tỉ lệ phát xung.
- `test_memory_off`: điểm khi tắt bộ nhớ.

## 1. Chuẩn bị

Cần có:

- Python 3.11 và Git.
- `ffprobe`, đi kèm bộ [FFmpeg](https://ffmpeg.org/download.html). Gõ `ffprobe -version` để kiểm tra, vì lệnh này phải chạy được từ terminal.
- Khoảng 9 GB trống: file nén 7 GB và video đã giải nén 1,9 GB.
- Không cần GPU để xem dữ liệu và chạy demo.

Tải code và tạo môi trường:

```bash
git clone https://github.com/incognito-tab/XLDL.git
cd XLDL
python -m venv .venv
```

Kích hoạt môi trường:

```bash
source .venv/Scripts/activate   # Git Bash trên Windows
source .venv/bin/activate       # macOS, Linux
```

```powershell
.venv\Scripts\Activate.ps1      # PowerShell trên Windows
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Mọi lệnh bên dưới đều chạy từ thư mục gốc của repo, với môi trường đã kích hoạt.

## 2. Tải dữ liệu UCF101

Dữ liệu không nằm trong git, nên mỗi người phải tự tải:

```bash
mkdir -p data/ucf101
curl -L -o data/ucf101/UCF-101.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip
curl -L -o data/ucf101/splits.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF101TrainTestSplits-RecognitionTask.zip
python -c "import zipfile; zipfile.ZipFile('data/ucf101/splits.zip').extractall('data/ucf101/splits')"
```

Trên PowerShell, thay `curl` bằng `curl.exe`, và dùng `mkdir data/ucf101 -Force` cho dòng đầu.

File `UCF-101.zip` phải có đúng `6957373664` byte. Nếu thiếu, chạy lại lệnh tải với `-C -` để tải tiếp:

```bash
curl -L -C - -o data/ucf101/UCF-101.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip
```

Đây là bản sao UCF101 trên Hugging Face, vì máy chủ chính thức rất chậm. File chia tập giống hệt bản chính thức, có cùng mã sha256 `5c0d1a53...bd27`. Nếu bạn đã có file chia tập từ [trang chính thức](https://www.crcv.ucf.edu/data/UCF101.php) thì dùng file đó cũng được.

## 3. Giải nén và kiểm tra

```bash
python -m vmem.prepare_ucf101 --root data/ucf101
python -m unittest tests.test_streams tests.test_models
```

Kết quả đúng:

- Lệnh đầu in `Wrote 3255 clips`.
- Lệnh sau kết thúc bằng `OK`, với 29 kiểm thử.

Các kiểm thử bao gồm:

- Không rò rỉ giữa các tập.
- Đáp án khớp với cách tính độc lập.
- Mô hình không nhìn trước tương lai.
- Các nơ-ron chỉ gửi xung cho nhau.

## 4. Chạy demo

Demo là một trang web chạy trên máy bạn:

1. Chọn luồng video test và bấm Phát.
2. Dừng lúc nào cũng được rồi đặt câu hỏi về quá khứ.
3. Xem đáp án đúng kèm bằng chứng, và câu trả lời của ba mô hình cùng độ tin cậy và thời gian chạy.

**Bước 1. Lấy file mô hình.** Demo cần thư mục `data/demo/` khoảng 220 MB. Thư mục này gồm đặc trưng ảnh đã tính sẵn và 3 mô hình đã huấn luyện. Có hai cách lấy:

- **Không có tài khoản Modal:** xin nhóm file `demo.zip` (47 MB) trong Google Drive của nhóm, rồi giải nén vào `data/`:

  ```bash
  python -c "import zipfile; zipfile.ZipFile('demo.zip').extractall('data')"
  ```

- **Có quyền vào Modal của nhóm:** tải thẳng từ volume `xldl-data`:

  ```bash
  mkdir -p data/demo && cd data/demo
  modal volume get xldl-data features/sresnet18_s0_w8.npy .
  modal volume get xldl-data frames112/index.json .
  modal volume get xldl-data memory/sresnet18_s0_w8/mixer_tau24_query_readTrue_hidden448_s0_e60_clock_rel_final.pt .
  modal volume get xldl-data memory/sresnet18_s0_w8/dmp_memory_size40_theta256_s0_e60_clock_rel_final.pt .
  modal volume get xldl-data memory/sresnet18_s0_w8/plus_channels16_memory_size10_thetas16-64-256_query_readTrue_delay_readTrue_s0_e60_clock_rel_final.pt .
  cd ../..
  ```

Kết quả đúng là `data/demo/` có 5 file: `sresnet18_s0_w8.npy`, `index.json` và 3 file `.pt`.

**Bước 2. Chạy server.** Cần đã làm xong bước 2 và 3 ở trên, vì demo đọc video từ `data/ucf101`.

```bash
python -m vmem.demo_server
```

Khi thấy dòng `Uvicorn running on http://127.0.0.1:8000`, mở địa chỉ đó bằng trình duyệt. Lần đầu mở một luồng mất vài giây để đọc video.

**Cách dùng:**

- **Chọn luồng:** có 40 luồng test, mỗi luồng 32 giây. Bấm Phát. Nút `1×` đổi tốc độ, thanh trượt dùng để tua.
- **Dòng thời gian:** chỉ hiện phần đã xem. Phần tương lai bị che sọc.
- **Đặt câu hỏi:** chọn loại câu hỏi, chỉnh N, T hoặc X, rồi bấm "Dừng và hỏi".
- **Câu hỏi có sẵn:** bấm để nhảy tới đúng lúc câu đó được hỏi. Nếu bật "Tự dừng ở câu hỏi có sẵn", video tự dừng và hỏi khi tới lúc.
- **Kết quả:** đáp án đúng có bằng chứng được viền vàng trên dòng thời gian. Mỗi mô hình hiện câu trả lời, đúng hay sai, top 3 kèm độ tin cậy, thời gian chạy trên CPU và tỉ lệ phát xung.
- **"Không hỏi được":** đây là đúng luật của bộ dữ liệu. Ví dụ thời điểm được hỏi nằm trong hoạt động đang chiếu, quá sát chỗ chuyển cảnh, hoặc X chưa xuất hiện.

Trước mỗi câu hỏi, đặc trưng của các khung hình sau thời điểm hỏi bị xóa, nên mô hình chỉ thấy quá khứ. Đặc trưng ảnh do bộ mã hóa SNN tính sẵn từ video thật, cho kết quả giống hệt khi mã hóa trực tiếp.

## 5. Xem dữ liệu

```bash
python -m vmem.build_data_viewer --root data/ucf101
```

Mở file `data/ucf101/viewer/index.html` bằng trình duyệt. Trang có:

- trình xem luồng;
- đáp án và bằng chứng của từng câu hỏi;
- thống kê 24 hoạt động, độ dài clip và cách chia tập.

Trang này và demo đều chứa khung hình UCF101, nên chỉ xem trên máy và đừng đưa lên mạng công khai.

## 6. Huấn luyện lại trên Modal (không bắt buộc)

Mọi lần huấn luyện đều chạy trên [Modal](https://modal.com). GPU dùng là L4 cho bộ mã hóa và T4 cho bộ nhớ.

Cài đặt và đăng nhập Modal:

```bash
pip install modal
modal token new
```

Dùng một trong hai cách:

- **Tài khoản của nhóm:** các bước nặng đã nằm sẵn trên volume.
- **Tài khoản riêng:** phải chạy lại từ đầu, gồm chuẩn bị dữ liệu, bộ mã hóa và đặc trưng. Việc này tốn vài giờ GPU.

Lệnh chính:

```bash
modal deploy modal_app.py                 # đưa code mới nhất lên Modal
modal run modal_app.py::round4            # 4 lần chạy của đợt 4
modal run modal_app.py::round4 --seeds 1,2   # thêm seed 1 và 2
```

Lần chạy nào đã có kết quả trên volume thì được đọc lại chứ không train lại. Mỗi lần chạy mới mất khoảng 30 phút trên T4. Kết quả được chép về `results/round4/`.

Các bước của đợt 3 và 4 đều được gói trong `modal_app.py`:

- `upgrade_pipeline`: cắt khung hình 112×112, huấn luyện Spiking ResNet18, rồi trích đặc trưng với `reset_every=8`.
- `round4`: các lần chạy cuối của đợt 4.

Đầu file `modal_app.py` liệt kê lệnh của các đợt trước.

**Lưu ý:** sau khi sửa code, phải chạy `modal deploy` trước khi gọi lại. Nếu không, Modal có thể vẫn chạy bản code cũ và trả về kết quả cũ.

### So sánh CNN và SNN

```bash
modal run modal_app.py::compare_cnn
modal run modal_app.py::compare_cnn --seeds 0,1,2
```

Lệnh train hai CNN nhân quả theo thời gian với đầu trả lời ReLU. Một bản có 8 tầng tích chập kernel 2, bản kia có 4 tầng kernel 4. Cả hai dùng 192 kênh, vùng tiếp nhận 256 frame và cùng đặc trưng Spiking ResNet18 `sresnet18_s0_w8` đã lưu trên volume. Các CNN được train 60 epoch theo thiết lập đợt 4. Các lần chạy đã hoàn thành trên volume được dùng lại.

So sánh điểm CNN trong `results/cnn_comparison/` với các điểm SNN trong `results/round4/` ở cùng seed. `test.mean_acc` là độ chính xác tổng thể, `test.by_kind` là độ chính xác từng loại câu và `test.by_age` cho biết khả năng nhớ theo độ xa. Phép so sánh bao gồm cả mô hình thời gian và đầu trả lời tương ứng. TCN không có neuron xung nên `firing_rate` là `null`.

### DualMemoryPlus ít tham số hơn

`memory_rank` giảm hạng của hai phép chiếu Legendre sang neuron LIF. `read_rank` giảm hạng của phép chiếu ở đầu đọc câu hỏi. Cả hai giữ nguyên số kênh, hệ số và cửa sổ nhớ. Có thể train từ đầu theo cùng thiết lập đợt 4 bằng `modal run modal_app.py::plus_compact --ranks 128,64`. Kết quả lưu trong `results/plus_compact/`.

## Dữ liệu gồm những gì

- **Nguồn:** 24 hoạt động trong [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), gồm 3.255 clip video thật.
- **Luồng video:** ghép các hoạt động dài ngẫu nhiên từ 2 đến 8 giây thành luồng 32 giây, lấy 8 khung hình/giây. Hoạt động có thể lặp lại, nhưng hai đoạn liền nhau luôn khác nhau.

| Tập | Nhóm UCF101 | Số clip |
|---|---|---:|
| Train | g11 đến g25 | 1.939 |
| Validation | g08 đến g10 | 398 |
| Test | g01 đến g07 | 918 |

Các clip cùng nhóm được cắt từ một video gốc, nên một nhóm chỉ được nằm trong một tập.

| Loại câu hỏi | Mức đoán bừa |
|---|---:|
| Cách đây N giây đang có hoạt động gì? | 4,2% |
| Ở giây thứ T đang xảy ra hoạt động gì? | 4,2% |
| Trong 4 hoạt động này, hoạt động nào đã xuất hiện? | 25% |
| Ngay sau lần gần nhất có X là hoạt động gì? | 4,2% |

Mỗi câu hỏi có đáp án kèm bằng chứng. Bằng chứng luôn nằm trong các đoạn đã kết thúc, nên muốn trả lời đúng thì mô hình phải nhớ.

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `vmem/classes.py` | 24 hoạt động và tên tiếng Việt |
| `vmem/prepare_ucf101.py` | Giải nén và ghi thông số từng clip |
| `vmem/streams.py` | Ghép luồng, sinh câu hỏi và mã hóa câu hỏi thành xung |
| `vmem/frames.py` | Cắt khung hình và ánh xạ luồng sang hàng đặc trưng |
| `vmem/encoder.py` | Bộ mã hóa ảnh SNN: sMLPNet của Mixer-SNN và Spiking ResNet18 |
| `vmem/train_encoder.py` | Huấn luyện bộ mã hóa và trích đặc trưng |
| `vmem/memory_models.py` | Ba mô hình bộ nhớ (Mixer, DMP, nâng cấp) và đầu trả lời câu hỏi |
| `vmem/train_memory.py` | Huấn luyện và chấm điểm mô hình bộ nhớ |
| `vmem/demo_server.py`, `vmem/demo.html` | Web demo |
| `vmem/build_data_viewer.py`, `vmem/viewer_template.html` | Trang xem dữ liệu |
| `modal_app.py` | Toàn bộ huấn luyện trên Modal |
| `tests/` | Kiểm thử dữ liệu và mô hình |
| `results/` | Kết quả của mọi lần chạy, dạng JSON |
| `models/`, `train.py`, `utils.py`, `samplers.py` | Mã gốc của Mixer-SNN, xem [docs/MIXER_SNN.md](docs/MIXER_SNN.md) |
| `Dual_memory_pathways/` | Mã gốc của DMP-SNN, dùng nguyên không sửa, xem [README riêng](Dual_memory_pathways/README.md) |

## Lỗi thường gặp

- **`FileNotFoundError` với `ffprobe`:** chưa cài FFmpeg, hoặc thư mục `bin` của FFmpeg chưa nằm trong `PATH`.
- **`No module named vmem`:** bạn đang đứng sai thư mục. Hãy chạy lệnh từ thư mục gốc của repo.
- **`No module named cv2`, `fastapi` hoặc `uvicorn`:** thư viện chưa được cài. Hãy kích hoạt môi trường rồi chạy lại `pip install -r requirements.txt`.
- **`BadZipFile` khi giải nén:** file zip chưa tải xong. Kiểm tra lại dung lượng ở bước 2.
- **Demo báo `missing checkpoints`:** `data/demo/` thiếu file `.pt`. Xem bước 1 của mục 4.
- **Demo báo `cannot read ... .avi`:** chưa giải nén video. Làm lại mục 3.
- **Cổng 8000 đã bị dùng:** chạy với cổng khác. Git Bash: `PORT=8001 python -m vmem.demo_server`. PowerShell: `$env:PORT=8001; python -m vmem.demo_server`.

Khi trích dẫn dữ liệu, dùng báo cáo [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402).
