import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import pytz
import altair as alt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="ARIMA BTC", page_icon="📈", layout="wide")
MOSCOW_TZ = pytz.timezone("Europe/Moscow")

# -------------------- UI --------------------
st.title("📈 ARIMA(4,1,1) — Bitcoin (CoinGecko)")
with st.sidebar:
    crypto = st.text_input("Криптовалюта (CoinGecko id)", value="bitcoin")
    train_period = st.selectbox("Период обучения (дней)", [30, 60, 90, 180, 365], index=3)
    steps = st.slider("Горизонт прогноза (дней)", 1, 14, 7)
    use_log = st.checkbox("Лог-трансформация (рекомендуется)", True)
    st.caption("Совет: держи горизонт 1–7 дней, чтобы избежать «выпрямления» на дальнем хвосте.")

# -------------------- Data --------------------
@st.cache_data(show_spinner=False, ttl=3600)
def load_coingecko_daily(coin_id: str, days: int = 730):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    px = data["prices"]
    df = pd.DataFrame(px, columns=["ts_ms", "close"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(MOSCOW_TZ).dt.date
    df = df[["date", "close"]].drop_duplicates("date").reset_index(drop=True)
    return df

def arima_forecast(y: np.ndarray, steps: int, order=(4,1,1), use_log=True) -> np.ndarray | None:
    if len(y) < max(30, order[0] + order[2] + 5):
        return None
    y = np.asarray(y, dtype=float)
    y_tr = np.log(y) if use_log else y
    model = ARIMA(y_tr, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(method="statespace", disp=0)
    fc = res.forecast(steps=steps)
    return np.exp(fc) if use_log else np.asarray(fc, dtype=float)

def backtest_metrics(y: np.ndarray, order=(4,1,1), use_log=True):
    # простая holdout-проверка: 80/20
    n = len(y)
    if n < 50:
        return None, None, None
    split = int(n * 0.8)
    train, test = y[:split], y[split:]
    if len(test) < 5:
        return None, None, None
    y_tr = np.log(train) if use_log else train
    model = ARIMA(y_tr, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(method="statespace", disp=0)
    fc = res.forecast(steps=len(test))
    fc = np.exp(fc) if use_log else np.asarray(fc, dtype=float)

    rmse = float(np.sqrt(np.mean((test - fc) ** 2)))
    mae = float(np.mean(np.abs(test - fc)))
    mape = float(np.mean(np.abs((test - fc) / (test + 1e-9))) * 100)
    return rmse, mae, mape

# -------------------- Run --------------------
try:
    df = load_coingecko_daily(crypto, days=max(365, train_period + 30))
    st.success(f"Загружено {len(df)} дневных точек")
except Exception as e:
    st.error(f"Не удалось загрузить данные: {e}")
    st.stop()

y_all = df["close"].values.astype(float)
y_train = y_all[-train_period:]

fc = arima_forecast(y_train, steps=steps, order=(4,1,1), use_log=use_log)
if fc is None:
    st.error("Недостаточно данных для ARIMA(4,1,1). Увеличь период обучения.")
    st.stop()

rmse, mae, mape = backtest_metrics(y_all, order=(4,1,1), use_log=use_log)

# -------------------- Metrics --------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Текущая цена", f"${y_all[-1]:,.2f}")
with col2:
    st.metric("Средний прогноз", f"${np.mean(fc):,.2f}")
with col3:
    st.metric("MAPE (holdout)", f"{mape:.1f}%" if mape is not None else "N/A")
with col4:
    st.metric("Модель", "ARIMA(4,1,1)")

# -------------------- Chart --------------------
hist_tail = min(100, len(y_all))
hist = pd.DataFrame({
    "idx": list(range(hist_tail)),
    "price": y_all[-hist_tail:],
    "type": ["История"] * hist_tail
})
future = pd.DataFrame({
    "idx": list(range(hist_tail-1, hist_tail-1 + steps)),
    "price": fc,
    "type": ["Прогноз"] * steps
})
chart_df = pd.concat([hist, future], ignore_index=True)

chart = alt.Chart(chart_df).mark_line(point=True).encode(
    x=alt.X("idx:Q", title="Период"),
    y=alt.Y("price:Q", title="USD", scale=alt.Scale(zero=False)),
    color=alt.Color("type:N", scale=alt.Scale(domain=["История", "Прогноз"], range=["#1f77b4", "#ff7f0e"])),
    tooltip=["type", "price"]
).properties(height=420).interactive()

st.altair_chart(chart, use_container_width=True)

st.caption(f"Обновлено: {datetime.now(MOSCOW_TZ).strftime('%Y-%m-%d %H:%M:%S')} МСК")
