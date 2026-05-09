import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cấu hình trang
st.set_page_config(page_title="GPA Predictor - NCKH", layout="wide")

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🎓 Dự án 4: App dự đoán GPA sinh viên")
st.markdown("### Theo thói quen học tập (Tập NCKH)")

# Sidebar: Tải dữ liệu
st.sidebar.header("Cấu hình dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file khảo sát (CSV hoặc Excel)", type=["csv", "xlsx"])

# Hàm làm sạch dữ liệu (như đã thảo luận ở bước trước)
def preprocess_data(df):
    # Mapping tên cột (Bạn hãy chỉnh sửa key cho khớp với file thực tế của bạn)
    mapping = {
        'Mỗi tuần, bạn thường dành ra bao nhiêu giờ để học? - Chỉ ghi số': 'So_gio_hoc_tuan',
        'Hiện tại bạn đang học bao nhiêu môn? - Chỉ ghi số': 'So_mon_hoc',
        'Bạn có đang làm thêm không?': 'Part_time',
        'Mỗi ngày, bạn thường ngủ bao nhiêu giờ?': 'Thoi_gian_ngu',
        'Bạn có đang tham gia bất kỳ CLB nào không?': 'Tham_gia_CLB',
        'Phần trăm tham gia các buổi học trên trường của bạn khoảng bao nhiêu?': 'Attendance_percent',
        'Bạn hay thường học theo cách nào?': 'Hoc_nhom',
        'Mỗi ngày, bạn thường sử dụng mạng xã hội bao lâu?': 'Social_media_time',
        'GPA hiện tại của bạn là bao nhiêu? (Theo thang 4.0)': 'GPA'
    }
    df = df.rename(columns=mapping)
    
    # Giữ lại các cột cần thiết
    required_cols = list(mapping.values())
    df = df[[c for c in required_cols if c in df.columns]]
    
    # Chuyển đổi số và xử lý Missing Values
    for col in df.select_dtypes(include=['object']).columns:
        if col in ['Part_time', 'Tham_gia_CLB', 'Hoc_nhom']:
            df[col] = df[col].apply(lambda x: 1 if 'có' in str(x).lower() or 'nhóm' in str(x).lower() else 0)
        else:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    df = df.dropna().reset_index(drop=True)
    return df

if uploaded_file is not None:
    # Đọc file
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    data = preprocess_data(df_raw)
    
    if len(data) < 10:
        st.error("Dữ liệu sau khi làm sạch quá ít (dưới 10 dòng). Vui lòng kiểm tra lại file đầu vào!")
    else:
        st.success(f"✅ Đã tải dữ liệu thành công! (Số mẫu: {len(data)})")
        
        # CHIA TAB: PHÂN TÍCH & DỰ ĐOÁN
        tab1, tab2, tab3 = st.tabs(["📊 Phân tích dữ liệu", "🤖 Huấn luyện mô hình", "🚀 Dự đoán thời gian thực"])
        
        with tab1:
            st.subheader("Ma trận tương quan (Correlation Heatmap)")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            
            st.warning("**Điểm học thuật:** Lưu ý hiện tượng 'Bias & self-reported data'. Sinh viên thường có xu hướng khai tốt hơn thực tế về số giờ học.")

        # Huấn luyện mô hình
        X = data.drop('GPA', axis=1)
        y = data['GPA']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        with tab2:
            col1, col2 = st.columns(2)
            col1.metric("R-Squared Score", f"{r2_score(y_test, y_pred):.3f}")
            col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.3f}")
            
            st.subheader("Mức độ ảnh hưởng của các yếu tố (Feature Importance)")
            importance = pd.DataFrame({'Feature': X.columns, 'Weight': model.coef_}).sort_values(by='Weight')
            fig_imp, ax_imp = plt.subplots()
            sns.barplot(x='Weight', y='Feature', data=importance, palette='viridis', ax=ax_imp)
            st.pyplot(fig_imp)

        with tab3:
            st.subheader("Nhập thông số của bạn để dự đoán GPA")
            c1, c2 = st.columns(2)
            with c1:
                in_study = st.number_input("Số giờ học/tuần", 0.0, 100.0, 20.0)
                in_courses = st.number_input("Số môn đang học", 1, 15, 5)
                in_parttime = st.selectbox("Làm thêm?", ["Không", "Có"])
                in_sleep = st.number_input("Thời gian ngủ (giờ/ngày)", 0.0, 24.0, 7.0)
            with c2:
                in_club = st.selectbox("Tham gia CLB?", ["Không", "Có"])
                in_attendance = st.slider("Tỷ lệ đi học (%)", 0, 100, 90)
                in_style = st.selectbox("Hình thức học", ["Tự học", "Học nhóm"])
                in_social = st.number_input("Dùng MXH (giờ/ngày)", 0.0, 24.0, 3.0)
            
            # Chuyển đổi input về dạng số
            input_features = pd.DataFrame([{
                'So_gio_hoc_tuan': in_study,
                'So_mon_hoc': in_courses,
                'Part_time': 1 if in_parttime == "Có" else 0,
                'Thoi_gian_ngu': in_sleep,
                'Tham_gia_CLB': 1 if in_club == "Có" else 0,
                'Attendance_percent': in_attendance,
                'Hoc_nhom': 1 if in_style == "Học nhóm" else 0,
                'Social_media_time': in_social
            }])
            
            if st.button("Dự đoán ngay"):
                prediction = model.predict(input_features)[0]
                prediction = np.clip(prediction, 0.0, 4.0)
                st.markdown(f"## Kết quả dự đoán GPA: `{prediction:.2f}`")
                if prediction >= 3.2: st.balloons()

else:
    st.info("Vui lòng tải file dữ liệu khảo sát ở Sidebar để bắt đầu.")