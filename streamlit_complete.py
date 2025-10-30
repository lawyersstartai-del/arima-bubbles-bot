import streamlit as st
import numpy as np
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# ------------------- НАСТРОЙКИ -------------------
st.set_page_config(page_title="ARIMA BTC", page_icon="📈", layout="wide")

# ВСТАВЬ СВОИ ДАННЫЕ:
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID"

SEND_PHOTO_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
# -------------------------------------------------

st.title("ARIMA прогноз BTC + Matplotlib + Telegram")

@st.cache_data(ttl=3600, show_spinner=False)
def load_btc_daily(days: int = 400) -> np.ndarray:
    """Загрузка дневных цен BTC в USD с CoinGecko."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    prices = [p[1] for p in data["prices"]]
    return np.array(prices, dtype=float)

# --------- ПАРАМЕТРЫ UI ----------
train_days   = st.slider("Период обучения (дней)", 30, 365, 120)
forecast_h   = st.slider("Прогноз вперёд (дней)", 1, 14, 7)
arima_order  = st.text_input("Порядок ARIMA (p,d,q)", "4,1,1")
use_log      = st.checkbox("Лог-трансформация (рекомендуется)", True)
# ----------------------------------

# Парсинг порядка
try:
    p, d, q = map(int, arima_order.split(","))
except Exception:
    st.error("Неверный формат порядка ARIMA. Пример: 4,1,1")
    st.stop()

# Данные
series_all = load_btc_daily(max(365, train_days + 30))
series = series_all[-train_days:]

st.write(f"Текущая цена: ${series_all[-1]:,.2f}")

# ---------- МОДЕЛЬ ARIMA ----------
def fit_forecast_arima(y: np.ndarray, steps: int, order: tuple[int,int,int], log: bool) -> np.ndarray:
    """Подгонка ARIMA и прогноз out-of-sample."""
    y_in = np.log(y) if log else y
    model = ARIMA(y_in, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit()
    fc = res.forecast(steps=steps)
    if log:
        fc = np.exp(fc)
    return np.asarray(fc, dtype=float)

forecast = fit_forecast_arima(series, forecast_h, (p, d, q), use_log)

# ---------- ГРАФИК В matplotlib ----------
fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
ax.plot(range(len(series)), series, label="История", color="#1f77b4")
ax.plot(range(len(series)-1, len(series)-1+forecast_h+1),
        np.concatenate([[series[-1]], forecast]),
        label="Прогноз", color="#ff7f0e")
ax.set_title(f"BTC/USD — ARIMA{(p,d,q)} — прогноз {forecast_h} дн.")
ax.set_xlabel("Дни")
ax.set_ylabel("USD")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig, use_container_width=True)

# ---------- ТАБЛИЦА ПРОГНОЗА ----------
tbl = pd.DataFrame({
    "День +": np.arange(1, forecast_h+1),
    "Прогноз, USD": forecast,
    "Δ% к текущей": (forecast / series_all[-1] - 1.0) * 100.0
})
st.dataframe(tbl.style.format({"Прогноз, USD": "{:,.2f}", "Δ% к текущей": "{:+.2f}%"}), use_container_width=True)

# ---------- ОТПРАВКА В TELEGRAM ----------
def send_plot_to_telegram(fig, chat_id: str, caption: str = "") -> bool:
    """Отправляет изображение графика как фото в Telegram."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    files = {"photo": ("arima.png", buf, "image/png")}
    data = {"chat_id": chat_id, "caption": caption}
    try:
        r = requests.post(SEND_PHOTO_URL, data=data, files=files, timeout=30)
        return r.ok
    except Exception:
        return False

caption = f"ARIMA{(p,d,q)} BTC/USD\\nТекущая: ${series_all[-1]:,.2f}\\nГоризонт: {forecast_h} дн."
if st.button("📤 Отправить график в Telegram"):
    ok = send_plot_to_telegram(fig, TELEGRAM_CHAT_ID, caption)
    if ok:
        st.success("Отправлено в Telegram")
    else:
        st.error("Не удалось отправить. Проверь токен/чат и интернет.")

st.caption(f"Обновлено: {datetime.now():%Y-%m-%d %H:%M:%S}")
