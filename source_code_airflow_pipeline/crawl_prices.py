import os
import pandas as pd
import requests
from datetime import date
from tqdm import tqdm
import time
import sys

# ==== CẤU HÌNH TOÀN CỤC ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "Price_update")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONSUMER_ID = "***"
CONSUMER_SECRET = "***"
FROM_DATE = "01/01/2015"
TO_DATE = date.today().strftime("%d/%m/%Y")


# ==== TOKEN ====
def get_access_token(consumer_id, consumer_secret):
    url = "https://fc-data.ssi.com.vn/api/v2/Market/AccessToken"
    payload = {
        "consumerID": consumer_id,
        "consumerSecret": consumer_secret
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()['data']['accessToken']


# ==== LẤY SYMBOL VN30 ====
def get_vn30_symbols(access_token):
    url = "https://fc-data.ssi.com.vn/api/v2/Market/IndexComponents"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"indexCode": "VN30", "pageIndex": 1, "pageSize": 50}
    response = requests.get(url, headers=headers, params=payload)
    response.raise_for_status()
    return [item['StockSymbol'] for item in response.json()['data'][0]['IndexComponent']]


# ==== API DỮ LIỆU GIÁ CÓ RETRY ====
def get_daily_ohlc_with_retry(access_token, symbol, from_date, to_date, max_retry=3, wait_time=3):
    url = "https://fc-data.ssi.com.vn/api/v2/Market/DailyOhlc"
    headers = {"Authorization": f"Bearer {access_token}"}
    all_data = []
    page_index = 1

    for attempt in range(max_retry):
        try:
            while True:
                payload = {
                    "symbol": symbol,
                    "fromDate": from_date,
                    "toDate": to_date,
                    "pageIndex": page_index,
                    "pageSize": 100,
                    "ascending": False
                }
                response = requests.get(url, headers=headers, params=payload, timeout=10)
                response.raise_for_status()
                data = response.json().get('data', [])
                if not data:
                    break
                all_data.extend(data)
                if len(data) < 100:
                    break
                page_index += 1

            if all_data:
                return all_data
            else:
                time.sleep(wait_time)
        except Exception as e:
            tqdm.write(f"❌ {symbol} attempt {attempt+1} lỗi: {e}")
            time.sleep(wait_time)
    return []


# ==== CORE CRAWL LOGIC ====
def crawl_vn30_prices():
    token = get_access_token(CONSUMER_ID, CONSUMER_SECRET)
    symbols = get_vn30_symbols(token)

    for symbol in tqdm(symbols, desc="Crawl VN30 symbols"):
        fname = os.path.join(OUTPUT_DIR, f"{symbol}_price.csv")

        # Bước 1: Ngày bắt đầu crawl
        if os.path.exists(fname):
            df_old = pd.read_csv(fname)
            df_old['TradingDate'] = pd.to_datetime(df_old['TradingDate'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
            df_old = df_old.dropna(subset=['TradingDate'])
            latest_date = df_old['TradingDate'].max()
            from_date_dynamic = (latest_date + pd.Timedelta(days=1)).strftime('%d/%m/%Y')
        else:
            df_old = None
            from_date_dynamic = FROM_DATE

        tqdm.write(f"📅 {symbol}: crawl từ {from_date_dynamic} → {TO_DATE}")

        # Bước 2: Crawl dữ liệu
        data = get_daily_ohlc_with_retry(token, symbol, from_date_dynamic, TO_DATE)

        if not data:
            tqdm.write(f"⚠️ {symbol}: không có dữ liệu mới.")
            with open(os.path.join(BASE_DIR, "../../missing_symbols_log.csv"), "a") as logf:
                logf.write(f"{symbol},{from_date_dynamic},{TO_DATE}\n")
            continue
        else:
            tqdm.write(f"📊 {symbol}: Lấy được {len(data)} dòng mới từ API.")

        # Bước 3: Làm sạch
        df = pd.DataFrame(data)
        df['TradingDate'] = pd.to_datetime(df['TradingDate'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
        df = df.dropna(subset=['TradingDate'])
        if 'Volume' not in df.columns:
            df['Volume'] = 0

        # Bước 4: Gộp dữ liệu
        if df_old is not None:
            df_combined = pd.concat([df_old, df], ignore_index=True)
            df_combined['TradingDate'] = pd.to_datetime(df_combined['TradingDate'], format='%d/%m/%Y', dayfirst=True, errors='coerce')
            df_combined = df_combined.dropna(subset=['TradingDate'])
            df_combined = df_combined.sort_values(['TradingDate', 'Volume'], ascending=[False, False])
            df_combined = df_combined.drop_duplicates(subset='TradingDate', keep='first')
            df_combined['TradingDate'] = df_combined['TradingDate'].dt.strftime('%d/%m/%Y')
        else:
            df['TradingDate'] = df['TradingDate'].dt.strftime('%d/%m/%Y')
            df_combined = df

        # Bước 5: Lưu file
        df_combined.to_csv(fname, index=False)
        tqdm.write(f"✅ {symbol}: tổng {len(df_combined)} dòng → {fname}")


    print("\n🎉 Hoàn tất cập nhật dữ liệu VN30!")


# ==== ĐIỂM ENTRY CHO DAG ====
def run():
    try:
        crawl_vn30_prices()
    except Exception as e:
        print(f"❌ Lỗi trong run(): {e}")
        sys.exit(1)


# ==== CHẠY THỬ LOCAL ====
if __name__ == "__main__":
    run()
