# map_to_fact_data.py

import os
import pandas as pd
import sys

# ==== Đường dẫn ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRICE_DIR = os.path.join(BASE_DIR, "..", "Price_update")
SENTIMENT_DIR = os.path.join(BASE_DIR, "..", "sentiment_results")
VN30_DATA_DIR = os.path.join(BASE_DIR, "..", "vn30_data")
FACT_DATA_PATH = os.path.join(BASE_DIR, "..", "fact_vn30_data.csv")


# ==== Đảm bảo thư mục đầu ra tồn tại ====
os.makedirs(VN30_DATA_DIR, exist_ok=True)

# ==== Hàm xử lý chính ====
def build_fact_data():
    all_data = []

    for fn in os.listdir(PRICE_DIR):
        if not fn.endswith("_price.csv"):
            continue

        symbol = fn.replace("_price.csv", "")
        price_path = os.path.join(PRICE_DIR, fn)
        sentiment_path = os.path.join(SENTIMENT_DIR, f"{symbol}_sentiment.csv")
        output_path = os.path.join(VN30_DATA_DIR, f"{symbol}_data.csv")

        if not os.path.exists(sentiment_path):
            print(f"⚠️ Bỏ qua {symbol}: không có file sentiment.")
            continue

        try:
            price_df = pd.read_csv(price_path)
            sentiment_df = pd.read_csv(sentiment_path)

            price_df['TradingDate'] = pd.to_datetime(price_df['TradingDate'], format='%d/%m/%Y', errors='coerce').dt.date
            sentiment_df['Time'] = pd.to_datetime(sentiment_df['Time'], errors='coerce').dt.date

            daily_sentiment = (
                sentiment_df.groupby("Time")["SentimentScore"]
                .mean().reset_index()
                .rename(columns={"Time": "TradingDate", "SentimentScore": "AvgSentimentScore"})
            )

            daily_sentiment["TradingDate"] = pd.to_datetime(daily_sentiment["TradingDate"]) + pd.Timedelta(days=1)
            daily_sentiment["TradingDate"] = daily_sentiment["TradingDate"].dt.date

            merged_df = price_df.merge(daily_sentiment, on="TradingDate", how="left")
            merged_df["AvgSentimentScore"] = merged_df["AvgSentimentScore"].fillna(0)
            merged_df = merged_df.sort_values("TradingDate", ascending=False).reset_index(drop=True)

            merged_df.to_csv(output_path, index=False)

            merged_df["Symbol"] = symbol
            all_data.append(merged_df)

            print(f"✅ {symbol}: đã tạo {symbol}_data.csv ({len(merged_df)} dòng)")
        except Exception as e:
            print(f"❌ Lỗi xử lý {symbol}: {e}")

    if all_data:
        fact_df = pd.concat(all_data, ignore_index=True)
        fact_df["TradingDate"] = pd.to_datetime(fact_df["TradingDate"])
        fact_df["DateKey"] = fact_df["TradingDate"].dt.strftime("%Y%m%d").astype(int)

        fact_df.to_csv(FACT_DATA_PATH, index=False)

        print(f"\n✅ Đã tạo bảng 'fact_vn30_data.csv' với {len(fact_df)} dòng.")
    else:
        print("⚠️ Không có dữ liệu để tạo bảng fact.")

# ==== Hàm để Airflow gọi ====
def run():
    try:
        build_fact_data()
    except Exception as e:
        print(f"❌ Lỗi trong run(): {e}")
        sys.exit(1)

# ==== Gọi khi chạy độc lập ====
if __name__ == "__main__":
    build_fact_data()
