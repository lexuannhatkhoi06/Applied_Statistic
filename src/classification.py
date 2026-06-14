"""
src/classification.py
Mô hình Phân loại (Classification).
Nhiệm vụ: Dự đoán khả năng lọt vào Podium (Top 3) của tay đua.
Sử dụng Logistic Regression (tuyến tính) và Random Forest (phi tuyến).
"""

import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_SEED, TEST_SIZE, RF_N_ESTIMATORS, RF_MAX_DEPTH, MODEL_DIR

def run_logistic_regression(X_train, X_test, y_train, y_test):
    """Huấn luyện và đánh giá mô hình Logistic Regression (Đơn giản, dễ giải thích)."""
    logger = logging.getLogger("F1_System")
    logger.info("Đang huấn luyện Logistic Regression...")

    # Khởi tạo và huấn luyện mô hình
    model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)

    # Dự đoán
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"[CLASSIFICATION] KẾT QUẢ LOGISTIC REGRESSION")
    print(f"Độ chính xác (Accuracy): {acc:.4f}")
    print("Báo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Không Podium', 'Podium']))
    print("="*50)

    return model

def run_random_forest(X_train, X_test, y_train, y_test):
    """Huấn luyện, đánh giá và lưu mô hình Random Forest Classifier (Phức tạp, chính xác cao)."""
    logger = logging.getLogger("F1_System")
    logger.info("Đang huấn luyện Random Forest Classifier...")

    # class_weight='balanced' giúp mô hình xử lý việc số lượng xe không lọt Podium áp đảo xe lọt Podium
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_SEED,
        class_weight='balanced' 
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print(f"[CLASSIFICATION] KẾT QUẢ RANDOM FOREST CLASSIFIER")
    print(f"Độ chính xác (Accuracy): {acc:.4f}")
    print("Báo cáo chi tiết:")
    print(classification_report(y_test, y_pred, target_names=['Không Podium', 'Podium']))
    print("="*50)

    # Lưu trọng số mô hình
    model_path = os.path.join(MODEL_DIR, 'rf_podium_classifier.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Đã lưu mô hình Random Forest tại: {model_path}")

    return model

def execute_classification_pipeline(df):
    """
    Hàm điều phối luồng Classification.
    Được gọi từ main.py để thực thi các mô hình phân loại.
    """
    logger = logging.getLogger("F1_System")
    logger.info("=== BẮT ĐẦU MÔ HÌNH PHÂN LOẠI (CLASSIFICATION) ===")

    # Lựa chọn các đặc trưng đầu vào (Features)
    features = ['grid', 'is_street', 'car_strength_index']
    X = df[features]
    
    # Lựa chọn biến mục tiêu (Target)
    y = df['is_podium']

    # Chia tập Train/Test
    # Tham số stratify=y đảm bảo tỷ lệ Podium/Không Podium ở tập Test giống hệt tập Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Chạy lần lượt 2 thuật toán để so sánh
    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_random_forest(X_train, X_test, y_train, y_test)

    logger.info("Hoàn thành module Classification.")

# Khối lệnh dùng để test độc lập file này
if __name__ == "__main__":
    from src.utils import setup_logger
    from src.data_pipeline import execute_data_pipeline
    from src.features import construct_feature_space

    setup_logger()
    df_raw = execute_data_pipeline()
    df_ready = construct_feature_space(df_raw)
    execute_classification_pipeline(df_ready)