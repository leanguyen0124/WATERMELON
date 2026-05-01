# Datathon 2026: Breaking Business Boundaries

Dự án này tập trung vào việc phân tích dữ liệu kinh doanh và dự báo doanh thu hằng ngày cho một doanh nghiệp thương mại điện tử ngành thời trang. Mục tiêu cốt lõi là giải quyết các thách thức về tăng trưởng hậu khủng hoảng 2019, tối ưu hóa dòng tiền và xây dựng mô hình dự báo chính xác cho nhiệm vụ Task 3 của cuộc thi Datathon 2026.

## 1. Tổng quan dự án

Dự án bao gồm hai phần chính:
- **Phân tích dữ liệu khám phá (EDA):** Nghiên cứu các chu kỳ tài chính, hành vi khách hàng và hiệu quả của các kênh marketing.
- **Mô hình hóa dự báo (Predictive Modeling):** Xây dựng mô hình Ridge Regression để dự báo Doanh thu (Revenue) và Giá vốn (COGS) theo ngày với độ chính xác cao ($R^2 \approx 0.76$).

## 2. Cấu trúc thư mục

```text
├── datathon-2026-round-1/     # Tập dữ liệu gốc (CSV files)
├── ML/              # Mã nguồn mô hình ML
│   ├── feature_ml.ipynb       # Notebook chính thực hiện Feature Engineering và Modeling
│   ├── dataset/               # Dữ liệu đã xử lý và file submission
│   └── models/                # Các báo cáo giải thích mô hình (SHAP-style)
├── EDA/                       # Mã nguồn phân tích EDA
│   ├── eda.py                 # Thư viện tổng hợp các hàm tính toán và vẽ biểu đồ
│   └── outputs/               # Kết quả biểu đồ và bảng tóm tắt
├── artifacts/                 # Các bản báo cáo phân tích chính thức
└── README.md                  # Hướng dẫn này
```

## 3. Hướng dẫn cài đặt

Dự án sử dụng Python 3.9+. Khuyến nghị sử dụng `uv` hoặc `venv` để quản lý môi trường.

```bash
# Cài đặt các thư viện cần thiết
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 4. Cách chạy dự án để tái lập kết quả

### Bước 1: Tạo các biểu đồ phân tích EDA
File `eda.py` chứa các hàm để vẽ lại các biểu đồ xuất hiện trong báo cáo dựa trên dữ liệu đã được tính toán.

1. Đảm bảo dữ liệu thô nằm trong thư mục `datathon-2026-round-1/`.
2. Chạy script để xuất các biểu đồ:
    ```bash
    python EDA/eda.py
    ```
   *Lưu ý: Script này được thiết kế để vẽ lại các biểu đồ từ bảng tóm tắt có sẵn. Các hàm tính toán thô cũng có sẵn trong file để bạn có thể gọi khi cần.*

### Bước 2: Thực hiện Feature Engineering và Huấn luyện mô hình
Mọi quy trình từ xử lý dữ liệu thô, tạo đặc trưng (Lags, Profiles, Seasonality) đến huấn luyện mô hình đều được tích hợp trong Notebook chính.

1. Truy cập thư mục `ML/`.
2. Mở và chạy toàn bộ các Cell trong file `feature_ml.ipynb`.
3. Sau khi chạy xong, kết quả dự báo sẽ được lưu tại `ML/dataset/submission.csv`.


## 5. Kết quả đạt được

- **Mô hình dự báo:** Đạt $R^2 = 0.7666$ cho doanh thu, nhận diện tốt các đỉnh nhu cầu mùa vụ.
- **Báo cáo phân tích:** Cung cấp 3 khuyến nghị chiến lược về Email Marketing, Bundle Pricing và Tối ưu hóa tồn kho dựa trên bằng chứng dữ liệu cụ thể.

---
**Thông tin kỹ thuật:** 
- Mô hình sử dụng: Ridge Regression.
- Chiến lược validation: Walk-forward forecasting (không leakage).
- Giải thích mô hình: Tích hợp phân rã đóng góp (SHAP-style) theo ngôn ngữ kinh doanh.
