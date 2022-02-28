## Phân tích cảm xúc (sentiment analysis)

Xây dựng mô hình phân loại cảm xúc người dùng. Sử dụng bộ dữ liệu Vietnamese Students’ Feedback Corpus (UIT-VSFC). Trong tập tin tải xuống có đính kèm file README.md hướng dẫn sử dụng dữ liệu.

Yêu cầu đầu vào: cho một câu phản hồi của dựa trên dữ liệu đã huấn luyện.

Yêu cầu đầu ra: đánh giá cảm xúc của câu đó theo 3 loại: tiêu cực (0), trung tính (1), tích cực (2)

Quy tắc viết chương trình: chương trình viết dưới dạng dòng lệnh, hỗ trợ 2 tham số bắt buộc là input (câu đầu vào) và result (file kết quả). Tên tập tin chạy là sentiment_analysis.py.
Ví dụ:
`python sentiment_analysis.py --input "Môn học rất bổ ích" --result sentiment.txt`

Trong đó: --input "Môn học rất bổ ích" là câu đầu vào, sentiment.txt chứa cảm xúc, kết quả là 0 hoặc 1 hoặc 2

Chú ý:

- Sử dụng ngôn ngữ Python;

- Được sử dụng các thư viện hỗ trợ trong quá trình xây dựng. Tuy nhiên không được sử dụng các hàm có sẵn trong các thư viện, nghĩa là phải tự xây dựng một mô hình do bản thân làm.
