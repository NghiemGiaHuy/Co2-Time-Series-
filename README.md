# CO₂ Time Series Prediction

Dự án phân tích chuỗi thời gian nồng độ CO₂ và dự đoán giá trị ở bước thời gian kế tiếp. Chương trình tiền xử lý dữ liệu, tạo đặc trưng trễ, huấn luyện và so sánh ba mô hình hồi quy, sau đó vẽ biểu đồ giá trị thực tế so với dự đoán của mô hình có RMSE thấp nhất.

## Nội dung repository

```text
.
├── co2.csv   # Dữ liệu chuỗi thời gian với hai cột: time, co2
├── main.py   # Tiền xử lý, tạo đặc trưng, huấn luyện, đánh giá và trực quan hóa
└── README.md
```

## Yêu cầu

- Python 3
- pandas
- matplotlib
- scikit-learn

## Cài đặt và chạy

Clone repository, cài các thư viện cần thiết rồi chạy script từ thư mục dự án để `main.py` tìm thấy `co2.csv`:

```bash
git clone https://github.com/NghiemGiaHuy/Co2-Time-Series-.git
cd Co2-Time-Series-
python -m pip install pandas matplotlib scikit-learn
python main.py
```

Khi chạy, chương trình in thông tin dữ liệu, các giá trị thiếu, thống kê mô tả và bảng so sánh mô hình; cuối cùng mở biểu đồ **Actual vs Predicted**. Cần môi trường có hỗ trợ hiển thị cửa sổ đồ họa để xem biểu đồ.

## Quy trình

1. Đọc `co2.csv`, chuyển `time` sang kiểu ngày giờ và sắp xếp dữ liệu theo thời gian.
2. Nội suy tuyến tính các giá trị thiếu trong cột `co2`.
3. Tạo 5 đặc trưng trễ (`co2_lag_1` đến `co2_lag_5`) và dùng giá trị CO₂ của bước kế tiếp làm mục tiêu dự đoán.
4. Chia dữ liệu theo thứ tự thời gian: 80% đầu để huấn luyện và 20% cuối để kiểm tra.
5. Huấn luyện và đánh giá Linear Regression, Decision Tree Regressor và Random Forest Regressor bằng MAE, RMSE và R². Bảng kết quả được sắp xếp theo RMSE tăng dần.
6. Vẽ giá trị thực tế và dự đoán trên tập kiểm tra cho mô hình có RMSE thấp nhất.

## Kết quả tham khảo

Các kết quả được ghi chú trong `main.py`:

| Mô hình | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.360164 | 0.465279 | 0.990917 |
| Random Forest | 5.628664 | 7.051154 | -1.086053 |
| Decision Tree | 6.299342 | 7.711379 | -1.494991 |

**Nhận xét:** Linear Regression vượt trội trên tập kiểm tra: MAE và RMSE thấp nhất, trong khi R² gần 1, cho thấy dự đoán khá sát với giá trị thực tế trên tập dữ liệu này. Random Forest và Decision Tree có sai số cao hơn nhiều và R² âm, tức là trong lần đánh giá này chúng dự đoán kém hơn một dự đoán cơ sở đơn giản bằng giá trị trung bình của tập kiểm tra. Vì vậy, Linear Regression được chọn để vẽ biểu đồ Actual vs Predicted. Kết quả này chỉ phản ánh cách chia dữ liệu và các đặc trưng hiện tại (5 giá trị trễ), không đảm bảo mô hình sẽ tổng quát tốt cho giai đoạn hoặc dữ liệu khác.

Các số liệu trên là kết quả tham khảo được ghi chú trong mã nguồn; kết quả chạy lại có thể khác tùy phiên bản thư viện và dữ liệu.

## Cấu trúc dữ liệu

`co2.csv` có hai cột:

- `time`: ngày ghi nhận dữ liệu.
- `co2`: giá trị CO₂; ô trống được nội suy trong bước tiền xử lý.

## Ghi chú

Phần biểu đồ khám phá ban đầu (chuỗi thời gian và histogram) hiện đang được comment trong `main.py`. Script chạy toàn bộ quy trình trực tiếp khi được thực thi; cấu hình phụ thuộc và bộ kiểm thử riêng chưa có trong repository.
