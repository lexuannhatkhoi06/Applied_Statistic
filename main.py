"""
main.py
ĐIỂM KHỞI CHẠY HỆ THỐNG (ENTRY POINT)
Dự án: Hệ thống Phân tích & Dự đoán Lợi thế xuất phát F1 (H2 Hypothesis)

Cách chạy: Mở Terminal/Command Prompt tại thư mục gốc và gõ: python main.py
"""

import warnings
import logging

# Tắt các cảnh báo màu đỏ (DeprecationWarning, Future warning) 
# của thư viện nền để console trông chuyên nghiệp và sạch sẽ hơn
warnings.filterwarnings('ignore')

# Import các module cốt lõi từ package src/
from src.utils import setup_logger, enforce_determinism
from src.data_pipeline import execute_data_pipeline
from src.features import construct_feature_space
from src.eda_analyzer import perform_eda_analysis
from src.classification import execute_classification_pipeline
from src.regression import execute_regression_pipeline

def main():
    # 1. Khởi tạo hệ thống (Initialization)
    logger = setup_logger()
    logger.info("="*60)
    logger.info(" BẮT ĐẦU HỆ THỐNG PHÂN TÍCH F1 - H2 HYPOTHESIS")
    logger.info("="*60)
    
    # Khóa hoàn toàn tính ngẫu nhiên của phần cứng
    enforce_determinism()
    
    try:
        # 2. Xử lý dữ liệu thô (Data Pipeline)
        logger.info("\n>>> BƯỚC 1: XỬ LÝ DỮ LIỆU THÔ (DATA PIPELINE)")
        df_raw = execute_data_pipeline()
        
        # 3. Kỹ sư đặc trưng (Feature Engineering)
        logger.info("\n>>> BƯỚC 2: TẠO ĐẶC TRƯNG TOÁN HỌC (FEATURE ENGINEERING)")
        df_ready = construct_feature_space(df_raw)
        
        # 4. Phân tích thống kê & Hình ảnh (EDA)
        logger.info("\n>>> BƯỚC 3: PHÂN TÍCH KHÁM PHÁ VÀ XUẤT BIỂU ĐỒ (EDA)")
        perform_eda_analysis(df_ready)
        
        # 5. Khởi chạy Mô hình Phân loại (Classification)
        logger.info("\n>>> BƯỚC 4: HUẤN LUYỆN MÔ HÌNH PHÂN LOẠI (CLASSIFICATION)")
        execute_classification_pipeline(df_ready)
        
        # 6. Khởi chạy Mô hình Hồi quy & Deep Learning (Regression)
        logger.info("\n>>> BƯỚC 5: MÔ HÌNH HỒI QUY VÀ DEEP LEARNING (REGRESSION)")
        execute_regression_pipeline(df_ready)
        
        logger.info("\n" + "="*60)
        logger.info(" THÀNH CÔNG! HỆ THỐNG ĐÃ HOÀN TẤT TOÀN BỘ CHU TRÌNH.")
        logger.info(" Vui lòng kiểm tra thư mục 'output/' để lấy biểu đồ và mô hình.")
        logger.info("="*60)
        
    except FileNotFoundError as fnf_err:
        logger.error(f"LỖI DỮ LIỆU: {fnf_err}")
        logger.error("Vui lòng đảm bảo các file CSV đã được đưa vào đúng thư mục data/raw/")
    except Exception as e:
        logger.error(f"LỖI HỆ THỐNG KHÔNG XÁC ĐỊNH: {e}")
        raise e

if __name__ == "__main__":
    main()