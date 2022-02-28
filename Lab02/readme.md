## Rút trích keyword (keyword extraction)

Cho một văn bản tiếng Việt, rút trích các từ khóa (keyword) đại diện cho văn bản đó.
Các bạn có thể tham khảo một trong 3 hướng tiếp cận LSA, LDA, NMF trong các minh họa để thực hiện tương tự cho tiếng Việt.

Quy tắc viết chương trình: chương trình viết dưới dạng dòng lệnh, hỗ trợ 2 tham số bắt buộc là input_file và result, tên tập tin chạy là keyword_extraction.py.

Ví dụ:
`python keyword_extraction.py --input_file text.txt --result output.txt`
Trong đó: `text.txt` là văn bản đầu vào (cần trích từ khóa), output chứa các từ khóa kết quả
