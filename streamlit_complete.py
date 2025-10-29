import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pytz

TELEGRAM_BOT_TOKEN = "5628451765:AAF3eghUBVePX-I_j3Rg2WvWKFGkx4u1F7M"
TELEGRAM_CHAT_ID = "204683255"
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

st.set_page_config(page_title="ARIMA Bot", page_icon="📊", layout="wide")
st.title("📊 ARIMA + Market Order Bubbles")
st.markdown("**Прогноз + Telegram отправка каждый час на XX:02**")

with st.sidebar:
    st.title("⚙️ Параметры")
    symbol = st.text_input("Символ", value="BTCUSDT")
    interval = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    forecast_steps = st.slider("Шагов прогноза", 3, 14, 7)
    days_history = st.slider("Дней истории", 7, 365, 30)
    st.divider()
    st.success("✅ Telegram подключен")
    st.info("⏰ Московское время (UTC+3)\n📤 Автоотправка: XX:02\n📊 ARIMA прогноз")

if 'last_send_hour' not in st.session_state:
    st.session_state.last_send_hour = -1
if 'messages_sent' not in st.session_state:
    st.session_state.messages_sent = []

def get_moscow_time():
    return datetime.now(MOSCOW_TZ)

def get_historical_klines(symbol, interval, days):
    try:
        import ccxt
        exchange = ccxt.binance()
        
        timeframe_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
        tf = timeframe_map.get(interval, '1h')
        
        all_candles = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        
        while since < exchange.milliseconds():
            candles = exchange.fetch_ohlcv(symbol, tf, since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Open time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.drop_duplicates(subset=['Open time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

def calculate_arima_forecast(prices, forecast_steps=7):
    """Простой ARIMA-подобный прогноз"""
    if len(prices) < 10:
        return None
    
    # Используем последние 20 значений для расчета тренда
    recent = prices[-20:]
    
    # Расчет простого тренда
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 2)  # Полином 2й степени
    poly = np.poly1d(coeffs)
    
    # Прогноз
    future_x = np.arange(len(recent), len(recent) + forecast_steps)
    forecast = poly(future_x)
    
    return forecast

def calculate_bubbles(df):
    df = df.copy()
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open'].replace(0, 1)) * 100
    
    df['Volume_EMA'] = df['Volume'].ewm(span=30, adjust=False).mean()
    df['Volume_STD'] = df['Volume'].rolling(window=30).std()
    
    df['Upper_Threshold'] = df['Volume_EMA'] + 1.0 * df['Volume_STD']
    df['Lower_Threshold'] = df['Volume_EMA'] + 0.5 * df['Volume_STD']
    
    df['Bubble_Type'] = 'None'
    
    for i in range(30, len(df)):
        current_volume = df.loc[i, 'Volume']
        current_price_change = df.loc[i, 'Price_Change_Pct']
        lower_threshold = df.loc[i, 'Lower_Threshold']
        
        if pd.notna(lower_threshold):
            if current_price_change < -0.05 and current_volume > lower_threshold:
                df.loc[i, 'Bubble_Type'] = 'Red'
            elif current_price_change > 0.05 and current_volume > lower_threshold:
                df.loc[i, 'Bubble_Type'] = 'Green'
    
    return df

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        response = requests.post(url, data=data, timeout=30)
        return response.json().get('ok', False)
    except:
        return False

def run_analysis(symbol, interval, forecast_steps, days_history):
    df = get_historical_klines(symbol, interval, days_history)
    
    if df is None or len(df) < 100:
        return False
    
    prices = df['Close'].values
    arima_forecast = calculate_arima_forecast(prices, forecast_steps)
    
    if arima_forecast is None:
        return False
    
    df_with_bubbles = calculate_bubbles(df)
    current_price = prices[-1]
    
    moscow_time = get_moscow_time()
    
    msg = f"<b>📊 ОТЧЁТ ARIMA + BUBBLES</b>\n"
    msg += f"<b>Время (МСК):</b> {moscow_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    msg += f"<b>{symbol}</b> | {interval}\n"
    msg += f"<b>Цена:</b> ${current_price:.2f}\n\n"
    
    msg += f"<b>📈 Прогноз на {forecast_steps} шагов:</b>\n"
    for i, price in enumerate(arima_forecast, 1):
        change = ((price - current_price) / current_price) * 100
        arrow = "📈" if change > 0 else "📉"
        msg += f"{arrow} {i}: ${price:.2f} ({change:+.2f}%)\n"
    
    msg += "\n"
    
    red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
    green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
    
    msg += f"<b>🔴🟢 ПУЗЫРИ:</b>\n"
    msg += f"🔴 Красные: {red_count} | 🟢 Зелёные: {green_count}\n\n"
    
    forecast_avg = np.mean(arima_forecast)
    if forecast_avg > current_price * 1.01:
        msg += "🎯 <b>ПОКУПКА</b> 📈\n"
    elif forecast_avg < current_price * 0.99:
        msg += "🎯 <b>ПРОДАЖА</b> 📉\n"
    else:
        msg += "⏳ <b>ОЖИДАНИЕ</b> ➡️\n"
    
    send_telegram_message(msg)
    return True

# MAIN
st.markdown("---")

moscow_time = get_moscow_time()
current_hour = moscow_time.hour
current_minute = moscow_time.minute

should_send = (current_minute == 2) and (st.session_state.last_send_hour != current_hour)

if should_send:
    with st.spinner("⏳ Отправляю отчёт..."):
        if run_analysis(symbol, interval, forecast_steps, days_history):
            st.session_state.last_send_hour = current_hour
            st.session_state.messages_sent.append(moscow_time)
            st.success(f"✅ Отчёт отправлен в {moscow_time.strftime('%H:%M:%S')} МСК")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🕐 Время (МСК)", moscow_time.strftime('%H:%M:%S'))
with col2:
    st.metric("📤 Отправлено", len(st.session_state.messages_sent))
with col3:
    st.metric("🤖 Статус", "🟢 РАБОТАЕТ")

st.markdown("---")
st.subheader("🚀 Ручная отправка")
if st.button("📤 ОТПРАВИТЬ ОТЧЁТ СЕЙЧАС", use_container_width=True, type="primary"):
    with st.spinner("⏳ Отправляю..."):
        if run_analysis(symbol, interval, forecast_steps, days_history):
            st.session_state.messages_sent.append(get_moscow_time())
            st.success("✅ Отчёт отправлен!")
        else:
            st.error("❌ Ошибка отправки")

st.markdown("---")
st.subheader("📊 Данные")

with st.spinner("⏳ Загружаю данные..."):
    df = get_historical_klines(symbol, interval, days_history)
    
    if df is not None and len(df) > 0:
        st.write("**📊 Последние 10 свечей:**")
        display_df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        display_df['Open time'] = display_df['Open time'].astype(str)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        df_with_bubbles = calculate_bubbles(df)
        red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
        green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
        
        st.write("**🔴🟢 Статистика:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Красные пузыри", red_count)
        with col2:
            st.metric("🟢 Зелёные пузыри", green_count)

st.markdown("---")
st.subheader("📤 История отправок")
if st.session_state.messages_sent:
    data = [{"Время (МСК)": t.strftime('%Y-%m-%d %H:%M:%S')} for t in st.session_state.messages_sent[-10:]]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

st.markdown("""<script>setTimeout(() => window.location.reload(), 60000);</script>""", unsafe_allow_html=True)
st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:11px;'>🤖 ARIMA Bubbles | CCXT + Telegram | Московское время (UTC+3)</div>", unsafe_allow_html=True)
