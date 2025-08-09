import os
import numpy as np
import pandas as pd
from datetime import timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import requests

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VN30_DATA_DIR = os.path.join(BASE_DIR, "..", "vn30_data")
FORECAST_PATH = os.path.join(BASE_DIR, "..", "vn30_lstm_tuned_forecast.csv")
EVALUATION_PATH = os.path.join(BASE_DIR, "..", "vn30_lstm_tuned_evaluation.csv")


def get_access_token(consumer_id, consumer_secret):
    url = "https://fc-data.ssi.com.vn/api/v2/Market/AccessToken"
    payload = {"consumerID": consumer_id, "consumerSecret": consumer_secret}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()['data']['accessToken']


def get_vn30_symbols(access_token):
    url = "https://fc-data.ssi.com.vn/api/v2/Market/IndexComponents"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"indexCode": "VN30", "pageIndex": 1, "pageSize": 50}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return [item["StockSymbol"] for item in response.json()['data'][0]['IndexComponent']]


def train_and_forecast_lstm(symbol, data_folder=VN30_DATA_DIR):
    print(f"🔄 Đang xử lý {symbol}...")
    file_path = os.path.join(data_folder, f"{symbol}_data.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ Không có dữ liệu cho {symbol}")
        return None, None

    df = pd.read_csv(file_path)
    df["TradingDate"] = pd.to_datetime(df["TradingDate"])
    df.sort_values("TradingDate", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["PriceChange"] = df["Close"] - df["Open"]
    df["High_Low_Spread"] = df["High"] - df["Low"]
    df["RollingMean_5"] = df["Close"].rolling(5).mean()
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["VolumeChange"] = df["Volume"].pct_change()

    features = [
        "Open", "High", "Low", "Close", "Volume", "Value", "AvgSentimentScore",
        "PriceChange", "High_Low_Spread", "RollingMean_5", "Momentum_10", "VolumeChange"
    ]
    df_features = df[features].dropna().copy()
    df_dates = df.loc[df_features.index, "TradingDate"].reset_index(drop=True)

    if len(df_features) < 200:
        print(f"⏭️ {symbol}: không đủ dữ liệu")
        return None, None

    std = StandardScaler()
    mm = MinMaxScaler()
    scaled = mm.fit_transform(std.fit_transform(df_features))
    close_idx = df_features.columns.get_loc("Close")

    def inverse_target(scaled_values):
        padded = np.zeros((scaled_values.shape[0], scaled.shape[1]))
        padded[:, close_idx] = scaled_values.flatten()
        return std.inverse_transform(mm.inverse_transform(padded))[:, close_idx]

    SEQ_LEN, HORIZON = 60, 1
    X, y = [], []
    for i in range(SEQ_LEN, len(scaled) - HORIZON):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i + HORIZON, close_idx])
    X, y = np.array(X), np.array(y)

    if len(X) < 200:
        print(f"⏭️ {symbol}: không đủ mẫu để train")
        return None, None

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    def objective(trial):
        units = trial.suggest_categorical("units", [32, 64])
        drop = trial.suggest_float("dropout", 0.2, 0.4, step=0.1)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = Sequential([
            LSTM(units, input_shape=(SEQ_LEN, X.shape[2])),
            Dropout(drop),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=3, factor=0.5, verbose=0)
        ]
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=25, batch_size=32, callbacks=callbacks, verbose=0)
        return min(hist.history["val_loss"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=6, show_progress_bar=False)
    best = study.best_params
    print(f"  ↪ {symbol}: Optuna chọn {best}")

    model = Sequential([
        LSTM(best["units"], input_shape=(SEQ_LEN, X.shape[2])),
        Dropout(best["dropout"]),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best["lr"]), loss="mse")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=25, batch_size=32,
              callbacks=[
                  EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
                  ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)
              ],
              verbose=0)

    y_pred = inverse_target(model.predict(X_test))
    y_true = inverse_target(y_test.reshape(-1, 1))
    mask = y_true != 0

    r2 = r2_score(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"📊 {symbol}: R²={r2:.4f} | RMSE={rmse:.2f} | MAE={mae:.2f} | MAPE={mape:.2f}%")

    evaluation_row = {
        "Symbol": symbol, "R2": round(r2, 4), "RMSE": round(rmse, 2),
        "MAE": round(mae, 2), "MAPE": round(mape, 2)
    }

    last_seq = scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, -1)
    df_dynamic = df_features.copy().reset_index(drop=True)
    forecast_prices = []
    forecast_dates = pd.bdate_range(start=df["TradingDate"].max() + timedelta(days=1), periods=30)

    for dt in forecast_dates:
        pred_scaled = model.predict(last_seq)[0][0]
        pred_real = float(inverse_target(np.array([[pred_scaled]]))[0])
        forecast_prices.append(pred_real)

        prev_row = df_dynamic.iloc[-1]
        open_new = prev_row["Close"]
        vol_prev = prev_row["Volume"]
        noise = np.random.uniform(0.98, 1.02)
        volume_new = vol_prev * noise

        value_new = prev_row["Value"]
        sentiment_new = prev_row["AvgSentimentScore"]
        high_new = max(open_new, pred_real)
        low_new = min(open_new, pred_real)
        close_new = pred_real
        price_change = close_new - open_new
        spread = high_new - low_new
        last_4 = df_dynamic["Close"].iloc[-4:].tolist()
        rolling5 = (sum(last_4) + close_new) / 5
        momentum10 = close_new - df_dynamic["Close"].iloc[-10] if len(df_dynamic) >= 10 else close_new - df_dynamic["Close"].iloc[0]
        vol_change = (volume_new - vol_prev) / vol_prev if vol_prev != 0 else 0.0

        new_row = pd.DataFrame([{
            "Open": open_new, "High": high_new, "Low": low_new, "Close": close_new,
            "Volume": volume_new, "Value": value_new, "AvgSentimentScore": sentiment_new,
            "PriceChange": price_change, "High_Low_Spread": spread,
            "RollingMean_5": rolling5, "Momentum_10": momentum10, "VolumeChange": vol_change
        }])
        df_dynamic = pd.concat([df_dynamic, new_row], ignore_index=True)

        new_scaled = mm.transform(std.transform(new_row))[0].reshape(1, 1, -1)
        last_seq = np.concatenate([last_seq[:, 1:, :], new_scaled], axis=1)

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Symbol": symbol,
        "Forecasted_Price": forecast_prices
    })

    return forecast_df, evaluation_row


def run_single_symbol(symbol, token=None):
    try:
        if token is None:
            token = get_access_token(
                consumer_id="dfe34cf59187473db8d719138d525791",
                consumer_secret="a1703f3842244b5d907c45dffd20e471"
            )
        forecast_df, evaluation_row = train_and_forecast_lstm(symbol)
        if forecast_df is not None and evaluation_row is not None:
            forecast_df.to_csv(os.path.join(VN30_DATA_DIR, f"{symbol}_forecast.csv"), index=False)
            pd.DataFrame([evaluation_row]).to_csv(os.path.join(VN30_DATA_DIR, f"{symbol}_evaluation.csv"), index=False)
            print(f"✅ Đã lưu kết quả cho {symbol}")
        else:
            print(f"⚠️ Bỏ qua {symbol} (thiếu dữ liệu)")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {symbol}: {e}")
    finally:
        tf.keras.backend.clear_session()


def run():
    token = get_access_token(
        consumer_id="dfe34cf59187473db8d719138d525791",
        consumer_secret="a1703f3842244b5d907c45dffd20e471"
    )
    symbols = get_vn30_symbols(token)

    forecast_all_df = pd.DataFrame(columns=["Date", "Symbol", "Forecasted_Price"])
    evaluation_df = pd.DataFrame(columns=["Symbol", "R2", "RMSE", "MAE", "MAPE"])

    for i, symbol in enumerate(symbols):
        run_single_symbol(symbol, token=token)

        forecast_path = os.path.join(VN30_DATA_DIR, f"{symbol}_forecast.csv")
        eval_path = os.path.join(VN30_DATA_DIR, f"{symbol}_evaluation.csv")

        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path)
            forecast_all_df = pd.concat([forecast_all_df, forecast_df], ignore_index=True)

        if os.path.exists(eval_path):
            eval_df = pd.read_csv(eval_path)
            evaluation_df = pd.concat([evaluation_df, eval_df], ignore_index=True)

        if (i + 1) % 5 == 0:
            print(f"🧹 Đã xử lý {i + 1} mã - Reset TensorFlow session...")
            tf.keras.backend.clear_session()

    forecast_all_df.to_csv(FORECAST_PATH, index=False)
    evaluation_df.to_csv(EVALUATION_PATH, index=False)
    print("\n✅ Đã lưu forecast và evaluation tổng hợp!")


if __name__ == "__main__":
    run()
