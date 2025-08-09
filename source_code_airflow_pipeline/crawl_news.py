import logging
import os
import random
import time
from time import sleep
import pandas as pd
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException,
    ElementClickInterceptedException, WebDriverException
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import sys

from .crawl_prices import get_vn30_symbols, get_access_token

# ========== CONFIG ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(BASE_DIR, "..", "News_update")
os.makedirs(NEWS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONSUMER_ID = "dfe34cf59187473db8d719138d525791"
CONSUMER_SECRET = "a1703f3842244b5d907c45dffd20e471"

# ========== DRIVER ==========
def setup_driver():
    options = uc.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(120)
    driver.implicitly_wait(10)
    return driver

# ========== LẤY NỘI DUNG BÀI VIẾT ==========
def get_article_content(driver):
    try:
        xpath = "/html/body/form/div[3]/div[2]/div[1]/table/tbody/tr[1]/td/table/tbody/tr[2]/td/table/tbody/tr[3]/td/table/tbody"
        elem = driver.find_element(By.XPATH, xpath)
        if elem.text.strip():
            return elem.text.strip()
    except NoSuchElementException:
        pass

    try:
        paras = driver.find_elements(By.CSS_SELECTOR, "div.detail-content p")
        return "\n".join(p.text for p in paras if p.text.strip())
    except:
        pass

    try:
        body_text = driver.find_element(By.TAG_NAME, "body").text.strip()
        lines = body_text.splitlines()
        return "\n".join([line for line in lines if len(line.strip()) > 30])
    except:
        return ""

# ========== CRAWL TIN TỨC 1 MÃ ==========
def scrape_news_for_stock(driver, stock):
    try:
        driver.get(f"https://s.cafef.vn/tin-doanh-nghiep/{stock}/event.chn")
    except WebDriverException as e:
        if 'tab crashed' in str(e):
            logging.warning(f"🔁 Tab crashed khi mở trang {stock}, sẽ bỏ qua mã này.")
            return
        else:
            raise e

    sleep(random.uniform(5, 8))

    time_list, title_list, link_list, content_list = [], [], [], []
    prev_links = set()

    save_path = os.path.join(NEWS_DIR, f"{stock}_news.csv")
    df_old = pd.read_csv(save_path) if os.path.exists(save_path) else pd.DataFrame(columns=['Time', 'Title', 'Link', 'Content'])
    today = pd.Timestamp.now().normalize()

    while True:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.docnhanhTitle"))
            )
            title_elems = driver.find_elements(By.CSS_SELECTOR, "a.docnhanhTitle")
            time_elems = driver.find_elements(By.CSS_SELECTOR, ".timeTitle")

            titles = [e.text.strip() for e in title_elems]
            links = [e.get_attribute("href") for e in title_elems]
            times = [e.text.strip() for e in time_elems]

            if not links or set(links).issubset(prev_links):
                logging.info("🛑 Không còn bài mới.")
                break

            prev_links.update(links)

            for i in range(len(times)):
                time_obj = pd.to_datetime(times[i], format="%d/%m/%Y %H:%M", errors='coerce')
                if pd.notna(time_obj) and time_obj.normalize() == today:
                    time_list.append(times[i])
                    title_list.append(titles[i])
                    link_list.append(links[i])

            try:
                next_btn = driver.find_element(By.ID, "aNext")
                driver.execute_script("arguments[0].click();", next_btn)
                sleep(random.uniform(2, 4))
            except (NoSuchElementException, TimeoutException, ElementClickInterceptedException):
                logging.info("⛔ Không tìm thấy nút chuyển trang.")
                break

        except Exception as e:
            logging.error(f"❌ Lỗi khi load trang tin {stock}: {e}")
            break

    for link in tqdm(link_list, desc=f"📄 {stock} - Crawling", unit="bài", leave=False):
        for attempt in range(3):
            try:
                driver.get(link)
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                content = get_article_content(driver)
                content_list.append(content)
                break
            except:
                if attempt == 2:
                    content_list.append("")
                sleep(random.uniform(1, 2))

    df_new = pd.DataFrame({
        "Time": time_list,
        "Title": title_list,
        "Link": link_list,
        "Content": content_list
    })

    df_new["Time"] = pd.to_datetime(df_new["Time"], format="%d/%m/%Y %H:%M", errors="coerce")
    df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset="Link")

    df_all['Time_tmp'] = pd.to_datetime(df_all["Time"], errors="coerce")
    df_all = df_all.sort_values("Time_tmp", ascending=False).drop(columns="Time_tmp")

    df_all.to_csv(save_path, index=False, encoding="utf-8-sig")
    logging.info(f"✅ {stock}: Lưu {len(df_all)} bài tại {save_path}")

# ========== CRAWL TOÀN BỘ VN30 ==========
def crawl_news_vn30():
    token = get_access_token(CONSUMER_ID, CONSUMER_SECRET)
    symbols = get_vn30_symbols(token)
    logging.info("📋 Danh sách mã VN30: %s", symbols)

    driver = setup_driver()
    try:
        for i, symbol in enumerate(symbols):
            if i > 0 and i % 5 == 0:  # 🔁 Restart driver mỗi 5 mã
                driver.quit()
                driver = setup_driver()
            logging.info(f"📰 Crawling news for {symbol}")
            scrape_news_for_stock(driver, symbol)
    finally:
        driver.quit()

# ========== GỌI TỪ AIRFLOW ==========
def run():
    try:
        crawl_news_vn30()
        logging.info("🏁 Crawl VN30 hoàn tất.")
    except Exception as e:
        logging.exception("❌ Lỗi trong run(): %s", e)
        sys.exit(1)

# ========== CHẠY LOCAL ==========
if __name__ == "__main__":
    run()
