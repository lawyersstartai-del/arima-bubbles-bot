import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import requests
import altair as alt
import os

# ----------------- ДИАГНОСТИКА ОКРУЖЕНИЯ -----------------
st.subheader("Диагностика окружения")
with st.expander("Показать установленные библиотеки"):
    st.code(os.popen('pip list').read())
# ---------------------------------------------------------

# Надёжный импорт ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    st.success("Statsmodels (новый путь) импортирован успешно!")
except ImportError:
    try:
        from statsmodels.tsa.arima_model import ARIMA
        st.success("Statsmodels (старый путь) импортирован успешно!")
    except ImportError as e:
        st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Statsmodels не найден. Проверь requirements.txt и перезагрузи приложение. {e}")
        st.stop()


MOSCOW_TZ = pytz.timezone("Europe/Moscow")

@st.cache_data(show_spinner=False, ttl=3600)
def load_coingecko_daily(coin_id, days=730):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        px = data["prices"]
        df = pd.DataFrame(px, columns=["ts_ms", "close"])
        df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(MOSCOW_TZ).dt.date
        df = df[["date", "close"]].drop_duplicates("date").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None

def arima_forecast(y, steps=7, order=(4,1,1), use_log=True):
    if len(y) < max(30, order[0] + order[2] + 5):
        return None
    y = np.asarray(y, dtype=float)
    y_tr = np.log(y) if use_log else y
    
    try:
        model = ARIMA(y_tr, order=order)
        res = model.fit()
        fc = res.forecast(steps=steps)
        return np.exp(fc) if use_log else np.asarray(fc, dtype=float)
    except Exception as e:
        st.warning(f"Ошибка при расчете ARIMA: {e}")
        return None

st.set_page_config(page_title="ARIMA BTC", page_icon="📈", layout="wide")
st.title("📈 ARIMA(4,1,1) — Bitcoin (CoinGecko)")
with st.sidebar:
    crypto = st.text_input("Криптовалюта (CoinGecko id)", value="bitcoin")
    train_period = st.selectbox("Период обучения (дней)", [30, 60, 90, 180, 365], index=3)
    steps = st.slider("Горизонт прогноза (дней)", 1, 14, 7)
    use_log = st.checkbox("Лог-трансформация (рекомендуется)", True)

df = load_coingecko_daily(crypto, days=max(365, train_period + 30))
if df is None:
    st.stop()

y_all = df["close"].values.astype(float)
y_train = y_all[-train_period:]

fc = arima_forecast(y_train, steps=steps, order=(4,1,1), use_log=use_log)
if fc is None:
    st.error("Недостаточно данных для ARIMA. Увеличь период обучения.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1: st.metric("Текущая цена", f"${y_all[-1]:,.2f}")
with col2: st.metric("Средний прогноз", f"${np.mean(fc):,.2f}")
with col3: st.metric("Модель", "ARIMA(4,1,1)")

hist_tail = min(100, len(y_all))
hist_df = pd.DataFrame({"idx": range(hist_tail), "price": y_all[-hist_tail:], "type": "История"})
forecast_df = pd.DataFrame({"idx": range(hist_tail - 1, hist_tail + steps), "price": np.insert(fc, 0, y_all[-1]), "type": "Прогноз"})

chart_df = pd.concat([hist_df, forecast_df])

chart = alt.Chart(chart_df).mark_line(point=True).encode(
    x=alt.X("idx:Q", title="Период"),
    y=alt.Y("price:Q", title="USD", scale=alt.Scale(zero=False)),
    color=alt.Color("type:N", scale=alt.Scale(domain=["История", "Прогноз"], range=["#1f77b4", "#ff7f0e"])),
    tooltip=["type", "price"]
).properties(height=420).interactive()

st.altair_chart(chart, use_container_width=True)
st.caption(f"Обновлено: {datetime.now(MOSCOW_TZ).strftime('%Y-%m-%d %H:%M:%S')} МСК")

