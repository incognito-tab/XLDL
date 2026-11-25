# Kịch bản thuyết trình

Đi kèm `docs/slides.html`, 27 slide, khoảng 15–18 phút. Mỗi mục dưới đây ứng với một slide.

Gợi ý chia người nói:
- **Thảo:** slide 1–11 (mở đầu, nhược điểm, bản nâng cấp) và 24–27 (demo, hạn chế, kết).
- **Huy:** slide 12–23 (dữ liệu, huấn luyện, kết quả).

---

## Slide 1 · Tiêu đề

Em chào thầy cô và các bạn. Nhóm em gồm Đặng Thị Ngọc Thảo và Đặng Nguyễn Hữu Huy. Đề tài của nhóm là xây dựng bộ nhớ video thời gian thực bằng mạng nơ-ron xung.

Cụ thể, nhóm em tìm ra nhược điểm của hai mô hình Mixer-SNN và DMP-SNN, rồi nâng cấp DMP-SNN để mô hình nhớ được những gì đã xảy ra trong video và trả lời câu hỏi về quá khứ.

## Slide 2 · Nội dung

Bài trình bày gồm 7 phần:
- nhược điểm của hai mô hình gốc;
- bản nâng cấp của nhóm;
- bộ dữ liệu;
- cách huấn luyện và thay đổi bộ mã hóa;
- kết quả;
- demo;
- hạn chế và hướng phát triển.

## Slide 3 · Bài toán dùng để đánh giá

Trước hết là bài toán mà nhóm dùng để so sánh. Mô hình xem một video dài 32 giây, từng khung hình một, giống như camera đang quay trực tiếp. Bất kỳ lúc nào mô hình cũng có thể bị hỏi về những gì đã qua.

Điểm quan trọng là mô hình không được xem lại khung hình cũ, nên bắt buộc phải tự nhớ. Toàn bộ mô hình là mạng nơ-ron xung.

Ví dụ ở giây 29,5 mô hình được hỏi: "Cách đây 29 giây đang có hoạt động gì?". Đáp án là dắt chó đi dạo, xảy ra từ giây 0 đến giây 2,9. Mô hình phải nhớ được chuyện từ gần 30 giây trước.

## Slide 4 · Mixer-SNN chỉ nhớ bằng điện thế màng

Nhược điểm đầu tiên thuộc về Mixer-SNN. Mixer-SNN không có bộ nhớ riêng. Thông tin quá khứ chỉ còn lại trong điện thế màng của nơ-ron, và điện thế màng rò rỉ theo hằng số thời gian τ.

Với τ = 2 như bản gốc, mô hình gần như quên ngay: chỉ đúng 11,7%, sát mức đoán bừa 9,4%. Nhóm đã dò nhiều giá trị τ để so sánh công bằng. Tăng τ thì giữ được lâu hơn, nhưng thông tin cũ và mới bị trộn lẫn. Tốt nhất là τ = 24, cũng chỉ đạt 23%. Tăng τ lên nữa thì lại giảm.

## Slide 5 · Mixer-SNN quên nhanh

Khi chia câu hỏi theo độ xa của sự kiện, nhược điểm này thấy rất rõ. Với sự kiện trong 4 giây gần nhất, Mixer đúng 58%. Sang 8 đến 16 giây thì chỉ còn 22%. Sau 16 giây chỉ còn 7,8%, thấp hơn cả đoán bừa.

Nhóm còn quan sát được một điều: khi cho câu hỏi đọc thẳng vào điện thế màng, lớp trên cùng của Mixer ngừng phát xung hoàn toàn. Mô hình chỉ còn dựa vào điện thế màng chứ không truyền thông tin bằng xung nữa.

## Slide 6 · DMP-SNN: bộ nhớ chỉ nhận một con số

DMP-SNN tốt hơn ở chỗ có bộ nhớ chậm Legendre, nhớ được lâu. Nhưng mỗi khung hình phải nén thành đúng 1 con số trước khi đưa vào bộ nhớ.

Với video, 1 con số không đủ để phân biệt 24 hoạt động. Vì vậy khi nhóm tăng kích thước bộ nhớ d từ 10 lên 40 rồi 80, kết quả gần như không đổi, thậm chí giảm ở d = 80. Nút thắt nằm ở đầu vào chứ không phải ở kích thước bộ nhớ. Ngoài ra DMP chỉ có một cửa sổ nhớ duy nhất.

## Slide 7 · Tóm lại

Tóm lại, Mixer-SNN nhớ bằng điện thế màng nên quên nhanh. Sau khi dò tham số, nó đạt 34,5% trên tập test, nhưng với sự kiện cách 16–32 giây chỉ còn 7,8%.

DMP-SNN nhớ được lâu hơn, nhưng thông tin bị nén quá mức nên nhớ mờ: tổng thể chỉ 28,4%. Từ hai nhược điểm này, nhóm thiết kế bản nâng cấp.

## Slide 8 · DualMemoryPlus: 4 thay đổi

Bản nâng cấp tên là DualMemoryPlus. Nhóm giữ bộ nhớ Legendre của DMP vì nó nhớ được lâu, và sửa 4 điểm:
- **H1:** bộ nhớ nhận một vector 16 kênh thay vì 1 con số, để giữ được nhiều thông tin hơn.
- **H2:** nhớ trên 3 cửa sổ cùng lúc, 2 giây, 8 giây và 32 giây, nên vừa nhớ chi tiết gần vừa nhớ xa.
- **H3:** câu hỏi được đọc thẳng vào bộ nhớ qua một lớp nơ-ron xung, nên mô hình biết cần lấy phần nào của quá khứ.
- **H4:** dùng đa thức Legendre để giải mã bộ nhớ đúng tại thời điểm được hỏi.

## Slide 9 · Kiến trúc

Khung hình đi qua bộ mã hóa Spiking ResNet18 để thành xung. Sau đó xung đi qua hai tầng nhớ, mỗi tầng có 16 kênh và 3 cửa sổ. Cuối cùng là đầu hỏi đáp bằng nơ-ron xung.

Câu hỏi cũng được mã hóa thành xung, gồm loại câu hỏi, số giây N, giây T, hoạt động X và đồng hồ hiện tại. Câu hỏi dùng để đọc bộ nhớ.

Nhóm có kiểm thử bảo đảm hai điều: mọi tín hiệu giữa các nơ-ron đều là xung, và mô hình không bao giờ nhìn thấy khung hình sau thời điểm được hỏi.

## Slide 10 · Mỗi thay đổi đóng góp bao nhiêu

Để biết thay đổi nào có tác dụng, nhóm thêm dần từng phần.

Với bộ mã hóa small, DMP gốc đạt 23,9%. Thêm H1 và H2 lên 32,6%. Thêm H3 lên 43,8%. Đây là bước tăng lớn nhất.

Với bộ mã hóa Spiking ResNet18, bản H1 đến H3 đạt 56,6%. Khi câu hỏi được viết lại theo khoảng cách tới hiện tại, kết quả lên 61,2%. Thêm H4 được 61,6%.

Khi tắt bộ nhớ, kết quả rơi xuống 9,6%, gần đoán bừa. Điều này chứng minh mô hình thật sự dựa vào bộ nhớ để trả lời.

## Slide 11 · Ưu điểm vượt trội

Tổng kết lại, bản nâng cấp có độ chính xác gấp 1,79 lần Mixer-SNN và 2,17 lần DMP-SNN. Với sự kiện cách 16–32 giây, nó vẫn đúng 63%, trong khi Mixer chỉ còn 7,8%. Nghĩa là mô hình gần như không quên theo thời gian.

Mô hình cũng gọn nhẹ:
- sau khi nén chỉ còn 0,55 triệu tham số;
- bộ nhớ chỉ có 960 con số, cố định dù video dài bao lâu;
- trả lời một câu hỏi mất khoảng 40 mili giây trên CPU.

## Slide 12 · Bộ dữ liệu UCF101

*(Huy)* Phần tiếp theo là bộ dữ liệu. Nhóm dùng UCF101, bộ video hành động thật quay từ YouTube. Nhóm chọn 24 hoạt động dễ phân biệt như thể thao, âm nhạc, sinh hoạt, tổng cộng 3.255 clip.

Các clip được chia theo nhóm gốc của UCF101. Clip cắt từ cùng một video chỉ nằm trong một tập, để mô hình không được thấy trước cảnh quay giống tập test.

## Slide 13 · Ghép thành luồng 32 giây

Các clip được ghép thành một luồng video 32 giây, 8 khung hình mỗi giây. Mỗi hoạt động kéo dài ngẫu nhiên từ 2 đến 8 giây.

Một hoạt động có thể lặp lại, như ví dụ trên slide có hai lần dắt chó đi dạo, nhưng hai đoạn liền nhau luôn khác nhau. Mỗi lần train, chương trình sinh ra luồng mới, nên mô hình không thể học thuộc.

## Slide 14 · 4 loại câu hỏi

Có 4 loại câu hỏi:
- cách đây N giây đang có hoạt động gì;
- ở giây thứ T đang xảy ra hoạt động gì;
- trong 4 hoạt động cho trước, hoạt động nào đã xuất hiện;
- ngay sau lần gần nhất có X là hoạt động gì.

Mỗi câu hỏi luôn có đáp án kèm bằng chứng, và bằng chứng luôn nằm trong đoạn đã kết thúc. Mô hình không thể trả lời bằng cách nhìn khung hình hiện tại mà bắt buộc phải nhớ. Mức đoán bừa trung bình là 9,4%.

## Slide 15 · Quy trình huấn luyện

Huấn luyện chia làm hai giai đoạn. Giai đoạn 1 train bộ mã hóa để nhận diện hoạt động trong từng giây video, rồi tính đặc trưng xung cho mọi clip một lần. Giai đoạn 2 train bộ nhớ trên các đặc trưng đó.

Để so sánh công bằng, ba mô hình dùng chung:
- đặc trưng, dữ liệu và đầu hỏi đáp;
- số bước cập nhật.

Nhóm còn dò tham số tốt nhất cho từng mô hình và giữ kích thước tương đương nhau, khoảng 0,8 triệu tham số.

Mô hình được chọn theo tập validation và chỉ chấm test một lần, trên 2.048 luồng. Mọi thứ được train trên GPU của Modal.

## Slide 16 · Lần 1: bộ mã hóa small

Lần đầu, nhóm dùng bộ mã hóa sMLPNet small của chính Mixer-SNN. Kết quả trung bình 3 seed:
- bản nâng cấp: 43,8%;
- Mixer: 28,8%;
- DMP: 23,9%.

Khi phân tích, nhóm thấy bộ mã hóa này chỉ nhận đúng 61,6% số clip. Nhóm tính thử: kể cả có bộ nhớ hoàn hảo, kết quả cũng chỉ đạt tối đa 58,8%. Như vậy nút thắt nằm ở bộ mã hóa, nên nhóm quyết định đổi.

## Slide 17 · Lần 2: Spiking ResNet18

Nhóm thay bằng Spiking ResNet18, khởi tạo từ ImageNet, dùng nơ-ron xung IF và ảnh 112×112. Bộ mã hóa mới nhận đúng 74,6% clip, và mức trần với bộ nhớ hoàn hảo tăng từ 58,8% lên 74,7%.

Một chi tiết quan trọng: nơ-ron IF tích lũy trạng thái. Khi train, bộ mã hóa chỉ thấy các đoạn 8 khung. Vì vậy khi trích đặc trưng, nhóm phải đặt lại trạng thái nơ-ron mỗi 8 khung cho khớp. Nếu không làm vậy, độ chính xác tụt xuống còn khoảng 50%.

## Slide 18 · Mô hình nào hưởng lợi nhiều nhất

Khi đổi bộ mã hóa, cả ba mô hình đều tăng, nhưng mức tăng rất khác nhau:
- bản nâng cấp tăng 12,8 điểm, từ 43,8% lên 56,6%;
- Mixer tăng 5,1 điểm;
- DMP chỉ tăng 2,2 điểm.

Ở lần này, cả ba mô hình đều được train 60 epoch và câu hỏi có thêm đồng hồ hiện tại. Kết quả cho thấy chỉ bản nâng cấp tận dụng được thông tin tốt hơn từ bộ mã hóa. Hai mô hình gốc bị giới hạn bởi chính cơ chế nhớ của chúng.

## Slide 19 · Kết quả cuối cùng

Đây là kết quả cuối cùng trên tập test:
- **bản nâng cấp:** 61,6%;
- **bản nâng cấp nén:** 61,2%, với 0,55 triệu tham số;
- **Mixer-SNN:** 34,5%;
- **DMP-SNN:** 28,4%.

Nhóm cũng đưa CNN tốt nhất vào để đối chiếu: CNN đạt 61,5% nhưng cần 2,85 triệu tham số. Phần này sẽ nói kỹ ở slide 23.

## Slide 20 · Theo loại câu hỏi

Bản nâng cấp dẫn đầu ở cả 4 loại câu hỏi. Chênh lệch lớn nhất nằm ở hai loại hỏi theo thời gian, "cách đây N giây" và "ở giây thứ T": bản nâng cấp đạt 65–69%, còn hai mô hình gốc chỉ 17–28%. Đây là tác dụng của bộ nhớ nhiều cửa sổ và cách đọc đúng thời điểm.

Câu "ngay sau X" khó nhất với mọi mô hình. Nhóm sẽ nói lý do ở phần hạn chế.

## Slide 21 · Nhớ xa không bị giảm

Bảng này cho thấy ưu điểm rõ nhất. Bản nâng cấp giữ được khoảng 60–65% dù sự kiện xảy ra 4 giây hay 30 giây trước. Mixer giảm từ 58% xuống 7,8%. DMP giảm từ 35% xuống 20%.

## Slide 22 · Giảm 34% tham số

Bạn Huy thử nén bản nâng cấp. Cách làm là tách các ma trận chiếu lớn thành hai ma trận nhỏ hạng 64 bằng SVD, không cần train lại. Kết quả:
- tham số giảm 34%, từ 831 nghìn xuống 552 nghìn;
- file mô hình từ 3,5 MB còn 2,4 MB;
- test chỉ giảm 0,47 điểm;
- với sự kiện xa, kết quả gần như giữ nguyên.

Điều này cho thấy mô hình còn có thể gọn hơn nữa mà không mất độ chính xác.

## Slide 23 · So với CNN

Nhóm cũng so sánh với mạng tích chập thường không dùng xung, dùng cùng đặc trưng và cùng cách train. CNN 192 kênh, có số tham số tương đương bản nâng cấp, chỉ đạt 48,8%. Tăng lên 384 hay 512 kênh còn kém hơn. Như vậy bài toán cần một cấu trúc nhớ phù hợp chứ không cần nhiều tham số.

Khi CNN được thêm bộ đệm lưu toàn bộ quá khứ, nó đạt 61,5%, ngang bản nâng cấp. Nhưng nó cần 2,85 triệu tham số, nhiều hơn 3,4 lần, hoặc 5,2 lần so với bản nén. Bộ đệm của nó lưu 16.384 số và tăng theo độ dài video. Bộ nhớ của bản nâng cấp chỉ 960 số và cố định. Ngoài ra SNN chỉ truyền xung, nên tiết kiệm năng lượng hơn khi chạy trên phần cứng nơ-ron xung.

## Slide 24 · Demo

*(Thảo)* Bây giờ nhóm em xin demo. *(Chuyển sang web demo.)*

Đây là luồng video test, mô hình chưa từng thấy khi train. Em sẽ cho video chạy, rồi dừng ở một thời điểm bất kỳ và đặt câu hỏi.

Bên phải hiện đáp án đúng kèm bằng chứng, được viền vàng trên dòng thời gian. Bên dưới là câu trả lời của ba mô hình, kèm độ tin cậy và thời gian chạy.

Các khung hình sau thời điểm hỏi đều bị xóa trước khi đưa vào mô hình, nên mô hình chỉ dựa vào những gì đã nhớ.

*Gợi ý khi demo:*
- hỏi một câu "cách đây N giây" với N lớn, khoảng 20–29 giây, để thấy Mixer quên còn bản nâng cấp vẫn đúng;
- bật "Tự dừng ở câu hỏi có sẵn" để video tự dừng và hỏi.

## Slide 25 · Vì sao mới đạt 61,6%

Vì sao mô hình mới đạt 61,6%? Lý do thứ nhất là bộ mã hóa chỉ nhận đúng khoảng 3/4 số clip, nên mức trần là 74,7%. Mô hình đã đạt 82% mức trần này. Phần lớn lỗi còn lại đến từ việc nhận diện sai hoạt động ngay từ đầu.

Lý do thứ hai là câu "ngay sau X" mới đạt 38,6%. Loại câu này cần nhớ thứ tự các sự kiện, mà bộ nhớ hiện tại nhớ theo thời gian chứ không theo sự kiện.

Cuối cùng, do giới hạn GPU, mỗi cấu hình cuối mới chạy 1 seed. Ở các lần chạy 3 seed trước đó, độ lệch chỉ khoảng 0,1–0,4 điểm.

## Slide 26 · Hướng phát triển

Nhóm đề xuất 4 hướng phát triển:
- **Bộ mã hóa tốt hơn:** thêm kênh chuyển động giống camera sự kiện, dùng mạng lớn hơn hoặc cửa sổ dài hơn để nâng mức trần.
- **Bộ nhớ theo sự kiện:** ghi lại mỗi lần hoạt động thay đổi, để trả lời tốt câu "ngay sau X".
- **Mã hóa trực tiếp:** nhận video bất kỳ từ camera hoặc file tải lên.
- **Phần cứng nơ-ron xung:** đưa mô hình lên chip neuromorphic để tận dụng ưu điểm tiết kiệm năng lượng.

## Slide 27 · Cảm ơn

Phần trình bày của nhóm em đến đây là hết. Cảm ơn thầy cô và các bạn đã lắng nghe. Nhóm em rất mong nhận được câu hỏi và góp ý.

---

## Câu hỏi có thể gặp

**Vì sao không dùng thẳng CNN, khi CNN có bộ đệm cũng đạt 61,5%?**
Hai bên ngang nhau về độ chính xác. Tuy vậy SNN có ít tham số hơn 3,4–5,2 lần và bộ nhớ cố định 960 số, còn bộ đệm của CNN tăng theo độ dài video. SNN chỉ truyền xung nên phù hợp với phần cứng tiết kiệm năng lượng. Đề tài cũng yêu cầu toàn bộ mô hình là SNN.

**So sánh có công bằng không?**
Ba mô hình dùng cùng đặc trưng, dữ liệu, đầu hỏi đáp và số bước cập nhật. Mỗi mô hình được dò tham số tốt nhất, và kích thước được cân bằng khoảng 0,8 triệu tham số. Mô hình được chọn theo validation, và test chỉ chấm một lần.

**Làm sao biết mô hình không nhìn trước tương lai?**
Có kiểm thử: thay đổi các khung hình sau thời điểm hỏi thì câu trả lời không đổi. Trong demo, các khung đó còn bị xóa hẳn.

**Làm sao biết mô hình thật sự dùng bộ nhớ?**
Khi tắt bộ nhớ, độ chính xác rơi từ 61,6% xuống 9,6%, gần mức đoán bừa.

**Kết quả 1 seed có đáng tin không?**
Ở đợt 3 seed trước đó, độ lệch giữa các seed chỉ 0,1–0,4 điểm. Khoảng cách giữa bản nâng cấp và hai mô hình gốc là 27–33 điểm, lớn hơn rất nhiều.
