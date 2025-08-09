import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from functools import lru_cache
import sys

# ==== Config ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(BASE_DIR, "..", "News_update")
SENTIMENT_DIR = os.path.join(BASE_DIR, "..", "sentiment_results")


os.makedirs(SENTIMENT_DIR, exist_ok=True)

# ==== Huggingface login ====
login(token="***")

@lru_cache()
def load_model():
    model_name = "phong02468/phobert-Vietnamese-newspaper-title-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = model.config.id2label
    return tokenizer, model, id2label

# ==== Phân tích cảm xúc ====
def predict_sentiment_long(text, max_chunk_tokens=256):
    if not isinstance(text, str) or text.strip() == "":
        return "unknown", 0.0

    tokenizer, model, id2label = load_model()

    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_chunk_tokens] for i in range(0, len(tokens), max_chunk_tokens)]

    results = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=max_chunk_tokens)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        label_id = torch.argmax(probs, dim=1).item()
        label = id2label[label_id]
        confidence = probs[0][label_id].item()
        results.append((label, confidence))

    if not results:
        return "unknown", 0.0

    labels = [r[0] for r in results]
    majority_label = max(set(labels), key=labels.count)
    avg_conf = sum([r[1] for r in results if r[0] == majority_label]) / labels.count(majority_label)
    return majority_label, round(avg_conf, 4)

# ==== Hàm chính ====
def run_sentiment_analysis():
    for filename in os.listdir(NEWS_DIR):
        if not filename.endswith("_news.csv"):
            continue

        symbol = filename.replace("_news.csv", "")
        input_path = os.path.join(NEWS_DIR, filename)
        output_path = os.path.join(SENTIMENT_DIR, f"{symbol}_sentiment.csv")

        try:
            df_news = pd.read_csv(input_path)
            df_news["Content"] = df_news["Content"].astype(str).str.strip()
            df_news["Link"] = df_news["Link"].astype(str).str.strip()

            if "Time" in df_news.columns:
                df_news["Time"] = pd.to_datetime(df_news["Time"], errors="coerce")

            if os.path.exists(output_path):
                df_sentiment = pd.read_csv(output_path)
                df_sentiment["Link"] = df_sentiment["Link"].astype(str).str.strip()
                if "Time" in df_sentiment.columns:
                    df_sentiment["Time"] = pd.to_datetime(df_sentiment["Time"], errors="coerce")
                    latest_time = df_sentiment.iloc[0]["Time"]
                else:
                    latest_time = pd.Timestamp("2000-01-01")
            else:
                df_sentiment = pd.DataFrame(columns=list(df_news.columns) + ["Sentiment", "Confidence", "SentimentScore"])
                latest_time = pd.Timestamp("2000-01-01")

            existing_links = set(df_sentiment["Link"])
            df_new = df_news[~df_news["Link"].isin(existing_links)].copy()
            df_new = df_new[df_new["Time"] > latest_time]

            print(f"🧠 {symbol}: {len(df_new)} bài mới cần xử lý (sau {latest_time.strftime('%d/%m/%Y %H:%M')})")

            if not df_new.empty:
                sentiments, confidences, scores = [], [], []

                for content in tqdm(df_new["Content"], desc=f"🔍 Sentiment: {symbol}", leave=False):
                    label, score = predict_sentiment_long(content)

                    if label.upper() == "POS":
                        sentiment_score = score
                    elif label.upper() == "NEG":
                        sentiment_score = -score
                    else:
                        sentiment_score = 0.0

                    sentiments.append(label)
                    confidences.append(score)
                    scores.append(sentiment_score)

                df_new["Sentiment"] = sentiments
                df_new["Confidence"] = confidences
                df_new["SentimentScore"] = scores

                df_sentiment = pd.concat([df_sentiment, df_new], ignore_index=True)
                df_sentiment.drop_duplicates(subset="Link", keep="first", inplace=True)

            if "Time" in df_sentiment.columns:
                df_sentiment.sort_values("Time", ascending=False, inplace=True)

            df_sentiment.to_csv(output_path, index=False)
            print(f"✅ {symbol}: Đã lưu {output_path}")


        except Exception as e:
            print(f"❌ Lỗi xử lý {filename}: {e}")

# ==== Để Airflow gọi ====
def run():
    try:
        run_sentiment_analysis()
    except Exception as e:
        print(f"❌ Lỗi trong run(): {e}")
        sys.exit(1)

# ==== Chạy thủ công ====
if __name__ == "__main__":
    run_sentiment_analysis()
