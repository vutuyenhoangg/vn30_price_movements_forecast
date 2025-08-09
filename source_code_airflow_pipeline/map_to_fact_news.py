# map_to_fact_news.py

import os
import pandas as pd
import sys

# ==== Cấu hình đường dẫn ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_DIR = os.path.join(BASE_DIR, "..", "sentiment_results")
FACT_NEWS_PATH = os.path.join(BASE_DIR, "..", "fact_vn30_news.csv")


# ==== Hàm xử lý toàn bộ file sentiment ====
def build_fact_news():
    files = [f for f in os.listdir(SENTIMENT_DIR) if f.endswith("_sentiment.csv")]
    dfs = []

    for file in files:
        symbol = file.split("_")[0]
        file_path = os.path.join(SENTIMENT_DIR, file)
        try:
            df = pd.read_csv(file_path)

            # Parse ngày
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
            df["Date"] = df["Time"].dt.date
            df["DateKey"] = df["Time"].dt.strftime("%Y%m%d").astype("Int64")
            df["Symbol"] = symbol

            dfs.append(df)
        except Exception as e:
            print(f"❌ Lỗi đọc file {file}: {e}")

    # Gộp lại
    if dfs:
        fact_news_df = pd.concat(dfs, ignore_index=True)
        fact_news_df.to_csv(FACT_NEWS_PATH, index=False, encoding="utf-8-sig")
        print(f"✅ Đã tạo bảng 'fact_vn30_news.csv' với {len(fact_news_df)} dòng.")

        
    else:
        print("⚠️ Không có file sentiment nào được xử lý.")

# ==== Cho phép gọi từ Airflow ====
def run():
    try:
        build_fact_news()
    except Exception as e:
        print(f"❌ Lỗi trong run(): {e}")
        sys.exit(1)

# ==== Chạy trực tiếp nếu gọi bằng terminal ====
if __name__ == "__main__":
    build_fact_news()
