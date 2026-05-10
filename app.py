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
# ĐỌC DỮ LIỆU ĐÃ LÀM SẠCH
# ==========================================
@st.cache_data 
def load_data():
    try:
        df = pd.read_csv("Cleaned_Data.csv")
        # Xóa cột index bị thừa nếu có trong quá trình lưu file CSV
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except FileNotFoundError:
        return None

data = load_data()

# Giao diện chính
st.title("🎓 GPA TECH 🤖")
st.markdown("### Hệ thống phân tích thói quen học tập")

if data is not None:
    # PHẦN 1: HUẤN LUYỆN MÔ HÌNH (Chạy ngầm ngay khi mở app)
    X = data.drop('GPA', axis=1)
    y = data['GPA']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # PHẦN 2: CHIA TABS HIỂN THỊ
    tab1, tab2, tab3 = st.tabs(["📊 Phân tích & Tương quan", "🤖 Hiệu suất Mô hình", "🚀 Dự đoán GPA thời gian thực"])

    with tab1:
        st.subheader("Ma trận tương quan (Correlation Heatmap)")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
        st.info("**Học thuật:** Biểu đồ này cho thấy mức độ liên quan giữa các thói quen và điểm GPA.")

    with tab2:
        col1, col2 = st.columns(2)
        col1.metric("R-Squared (Độ chính xác)", f"{r2_score(y_test, y_pred):.3f}")
        col2.metric("Lỗi MSE", f"{mean_squared_error(y_test, y_pred):.3f}")
        
        st.subheader("Mức độ ảnh hưởng của các thói quen (Feature Importance)")
        importance = pd.DataFrame({'Thói quen': X.columns, 'Trọng số': model.coef_}).sort_values(by='Trọng số')
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Trọng số', y='Thói quen', data=importance, palette='coolwarm', ax=ax_imp)
        st.pyplot(fig_imp)
        st.caption("Trọng số dương (+) giúp tăng GPA, trọng số âm (-) làm giảm GPA.")

    with tab3:
        st.subheader("Nhập thông số cá nhân")
        c1, c2 = st.columns(2)
        
        with c1:
            in_study = st.number_input("Số giờ học/tuần", 0.0, 100.0, 20.0)
            in_courses = st.number_input("Số môn đang học", 1, 15, 5)
            in_parttime = st.selectbox("Làm thêm (0: Không, 1: Có)", [0, 1])
            in_sleep = st.number_input("Thời gian ngủ (giờ/ngày)", 0.0, 24.0, 7.0)
        
        with c2:
            in_club = st.selectbox("Tham gia CLB (0: Không, 1: Có)", [0, 1])
            in_attendance = st.slider("Tỷ lệ đi học (%)", 0, 100, 90)
            in_style = st.selectbox("Cách học (0: Tự học, 1: Học nhóm)", [0, 1])
            in_social = st.number_input("Dùng mạng xã hội (giờ/ngày)", 0.0, 24.0, 3.0)
        
        # Tạo DataFrame cho dữ liệu nhập vào
        input_features = pd.DataFrame([{
            'So_gio_hoc_tuan': in_study,
            'So_mon_hoc': in_courses,
            'Part_time': in_parttime,
            'Thoi_gian_ngu': in_sleep,
            'Tham_gia_CLB': in_club,
            'Attendance_percent': in_attendance,
            'Hoc_nhom': in_style,
            'Social_media_time': in_social
        }])

        if st.button("Dự đoán GPA ngay"):
            # CHỖ FIX CHÍNH: Ép thứ tự cột của input_features khớp chính xác 100% với model
            input_features = input_features.reindex(columns=X.columns, fill_value=0)
            
            # Tiến hành dự đoán
            prediction = model.predict(input_features)[0]
            
            # Giới hạn kết quả trong khoảng 0.0 - 4.0
            prediction = np.clip(prediction, 0.0, 4.0)
            
            st.markdown(f"## Kết quả dự đoán GPA của bạn: `{prediction:.2f}`")
            
            if prediction >= 3.6:
                st.success("🌟 Xuất sắc! Duy trì thói quen này nhé.")
                st.balloons()
            elif prediction >= 3.2:
                st.info("👍 Giỏi! Bạn đang làm rất tốt.")
            elif prediction >= 2.5:
                st.warning("📚 Khá. Cần tập trung thêm vào các yếu tố có trọng số dương.")
            else:
                st.error("⚠️ Trung bình/Yếu. Cần điều chỉnh lại thói quen học tập.")
else:
    st.error("❌ Không tìm thấy file 'Cleaned_Data.csv'. Vui lòng đảm bảo file nằm cùng thư mục với app.py")