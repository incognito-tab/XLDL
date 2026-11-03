# Nhớ video theo thời gian thực bằng mạng nơ-ron xung

Dự án xây một mạng nơ-ron xung (SNN) xem video thật đang chạy và trả lời câu hỏi về những gì đã qua, không được xem lại khung hình cũ. Mô hình nâng cấp sẽ được so sánh với Mixer-SNN và DMP-SNN.

**Trạng thái hiện tại:** dữ liệu và bộ sinh câu hỏi đã xong, chưa huấn luyện mô hình. Vì vậy hiện chỉ có dữ liệu để xem, chưa có kết quả độ chính xác.

Kế hoạch đầy đủ nằm ở [docs/plan.html](docs/plan.html). Tải file về rồi mở bằng trình duyệt.

## 1. Chuẩn bị

Cần có:

- Python 3.11 và Git.
- `ffprobe`, đi kèm bộ [FFmpeg](https://ffmpeg.org/download.html). Gõ `ffprobe -version` để kiểm tra, vì lệnh này phải chạy được từ terminal.
- Khoảng 9 GB trống: file nén 7 GB và video đã giải nén 1,9 GB.

Tải code và cài thư viện:

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

Sau đó cài thư viện:

```bash
pip install -r requirements.txt
```

## 2. Tải dữ liệu

Dữ liệu không nằm trong git, nên mỗi người phải tự tải. Chạy từ thư mục gốc của repo:

```bash
mkdir -p data/ucf101
curl -L -o data/ucf101/UCF-101.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip
curl -L -o data/ucf101/splits.zip https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
python -c "import zipfile; zipfile.ZipFile('data/ucf101/splits.zip').extractall('data/ucf101/splits')"
```

Trên PowerShell, thay `curl` bằng `curl.exe`, và dùng `mkdir data/ucf101 -Force` cho dòng đầu.

File `UCF-101.zip` phải có đúng `6957373664` byte. Nếu thiếu, chạy lại lệnh tải với `-C -` để tải tiếp:

```bash
curl -L -C - -o data/ucf101/UCF-101.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip
```

File này là bản sao UCF101 trên Hugging Face, vì máy chủ chính thức rất chậm khi nhóm thử tải. Bước 3 đối chiếu từng clip với danh sách chia tập chính thức. Nếu bước đó báo lỗi hoặc không ra đúng số clip, xóa file zip và tải lại.

## 3. Giải nén và kiểm tra

```bash
python -m vmem.prepare_ucf101 --root data/ucf101
python -m unittest tests.test_streams
```

Kết quả đúng:

- Lệnh đầu in `Wrote 3255 clips`.
- Lệnh sau kết thúc bằng `OK`, với 16 kiểm thử.

## 4. Xem dữ liệu

```bash
python -m vmem.build_data_viewer --root data/ucf101
```

Sau đó mở file `data/ucf101/viewer/index.html` bằng trình duyệt. Trang có:

- **Trình xem luồng:** chọn một luồng video, bấm Phát, rồi chọn câu hỏi để nhảy đến lúc nó được hỏi.
- **Đáp án và bằng chứng:** bấm "Hiện đáp án và bằng chứng" để xem đáp án cùng các đoạn chứng minh. Bấm vào ảnh bằng chứng để nhảy đến đoạn đó.
- **Dòng thời gian:** viền xanh là đoạn đang chiếu, ô vàng là bằng chứng, ô mờ là tương lai mà mô hình chưa thấy khi được hỏi.
- **Thống kê:** 24 hoạt động, độ dài clip, cách chia tập và các loại câu hỏi.

Trang chứa khung hình của UCF101, nên chỉ xem trên máy và đừng đưa lên mạng công khai.

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
| `vmem/streams.py` | Ghép luồng và sinh câu hỏi |
| `vmem/build_data_viewer.py`, `vmem/viewer_template.html` | Tạo trang xem dữ liệu |
| `tests/test_streams.py` | Kiểm thử chống rò rỉ dữ liệu |
| `models/`, `train.py`, `utils.py`, `samplers.py` | Mã gốc của Mixer-SNN, xem [docs/MIXER_SNN.md](docs/MIXER_SNN.md) |
| `Dual_memory_pathways/` | Mã gốc của DMP-SNN, xem [README riêng](Dual_memory_pathways/README.md) |

## Lỗi thường gặp

- **`FileNotFoundError` với `ffprobe`:** chưa cài FFmpeg, hoặc thư mục `bin` của FFmpeg chưa nằm trong `PATH`.
- **`No module named vmem`:** bạn đang đứng sai thư mục. Hãy chạy lệnh từ thư mục gốc của repo.
- **`No module named cv2`:** thư viện chưa được cài. Hãy kích hoạt môi trường rồi chạy lại `pip install -r requirements.txt`.
- **`BadZipFile` khi giải nén:** file zip chưa tải xong. Kiểm tra lại dung lượng ở bước 2.

Khi trích dẫn dữ liệu, dùng báo cáo [UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild](https://arxiv.org/abs/1212.0402).
