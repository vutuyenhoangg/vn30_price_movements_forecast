from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

# Bật log debug nếu cần
logging.basicConfig(level=logging.INFO)

# Import các module xử lý từng bước từ pipeline_vn30
from pipeline_vn30 import (
    crawl_prices,
    crawl_news,
    sentiment_analysis,
    map_to_fact_news,
    map_to_fact_data,
    train_model
)

# ==== Cấu hình mặc định cho các task ====
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['vutuyenhoang88@gmail.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# ==== Định nghĩa DAG ====
with DAG(
    dag_id='vn30_data_pipeline',
    default_args=default_args,
    description='Pipeline crawl giá, tin tức, sentiment, mapping và dự báo VN30, sau đó upload lên Lakehouse',
    schedule_interval='0 10 * * *',  # Chạy hàng ngày lúc 17h VN
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    concurrency=1,
    tags=['vn30', 'stock', 'forecast', 'sentiment'],
) as dag:

    # 1. Crawl giá cổ phiếu VN30
    crawl_prices_task = PythonOperator(
        task_id='crawl_vn30_prices',
        python_callable=crawl_prices.run,
    )

    # 2. Crawl tin tức liên quan đến cổ phiếu VN30
    crawl_news_task = PythonOperator(
        task_id='crawl_news_vn30',
        python_callable=crawl_news.run,
    )

    # 3. Phân tích cảm xúc từ tin tức
    sentiment_task = PythonOperator(
        task_id='sentiment_analysis',
        python_callable=sentiment_analysis.run,
    )

    # 4. Mapping tin tức vào bảng fact
    map_news_task = PythonOperator(
        task_id='map_to_fact_news',
        python_callable=map_to_fact_news.run,
    )

    # 5. Mapping dữ liệu giá và cảm xúc thành bảng fact
    map_data_task = PythonOperator(
        task_id='map_to_fact_data',
        python_callable=map_to_fact_data.run,
    )

    # 6. Huấn luyện và dự báo với mô hình biGRU
    train_forecast_task = PythonOperator(
        task_id='train_forecast_LSTM',
        python_callable=train_model.run,
        execution_timeout=timedelta(hours=4),
    )


    # ==== Định nghĩa thứ tự pipeline ====
    crawl_prices_task >> crawl_news_task >> sentiment_task >> map_news_task >> map_data_task >> train_forecast_task
