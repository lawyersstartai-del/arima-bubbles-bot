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
st.markdown("**Прогноз + Telegram (Demo Data)**")

with st.sidebar:
    st.title("⚙️ Параметры")
    symbol = st.text_input("Символ", value="BTCUSDT")
    interval = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    forecast_steps = st.slider("Шагов прогноза", 3, 14, 7)
    st.divider()
    st.success("✅ Telegram подключен")
    st.info("⏰ Московское время (UTC+3)\n📤 Demo API")

if 'last_send_hour' not in st.session_state:
    st.session_state.last_send_hour = -1
if 'messages_sent' not in st.session_state:
    st.session_state.messages_sent = []

def get_moscow_time():
    return datetime.now(MOSCOW_TZ)

def generate_demo_data(symbol, num_bars=1000):
    """Генерируем качественные демо-данные"""
    np.random.seed(42)
    
    base_price = 42000
    
    # Генерируем цены
    prices = [base_price]
    for i in range(num_bars):
        change = np.random.normal(0, 100)
        prices.append(max(prices[-1] + change, 1000))
    
    # Убеждаемся что у нас ровно num_bars свечей
    times = pd.date_range(end=datetime.now(), periods=num_bars, freq='1h')
    
    data = {
        'Open time': times,
        'Open': prices[:-1],
        'Close': prices[1:],
    }
    
    # Добавляем High и Low
    opens = np.array(prices[:-1])
    closes = np.array(prices[1:])
    
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 100, num_bars))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 100, num_bars))
    
    data['High'] = highs
    data['Low'] = lows
    data['Volume'] = np.random.uniform(1000, 100000, num_bars)
    
    df = pd.DataFrame(data)
    return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

def calculate_arima_forecast(prices, forecast_steps=7):
    if len(prices) < 10:
        return None
    
    recent = prices[-20:]
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 2)
    poly = np.poly1d(coeffs)
    
    future_x = np.arange(len(recent), len(recent) + forecast_steps)
    forecast = poly(future_x)
    
    return forecast

def calculate_bubbles(df):
    df = df.copy()
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open'].replace(0, 1)) * 100
    
    df['Volume_EMA'] = df['Volume'].ewm(span=30, adjust=False).mean()
    df['Volume_STD'] = df['Volume'].rolling(window=30).std()
    
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
        params = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, params=params, timeout=30)
        result = response.json()
        
        return result.get('ok', False)
    except:
        return False

def run_analysis(symbol, interval, forecast_steps):
    try:
        df = generate_demo_data(symbol)
        
        if len(df) < 100:
            st.error("❌ Недостаточно данных")
            return False
        
        prices = df['Close'].values.astype(float)
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
        
        msg += f"<b>📈 Прогноз:</b>\n"
        for i, price in enumerate(arima_forecast[:5], 1):
            change = ((price - current_price) / current_price) * 100
            arrow = "📈" if change > 0 else "📉"
            msg += f"{arrow} {i}: ${price:.2f} ({change:+.2f}%)\n"
        
        msg += "\n"
        
        red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
        green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
        msg += f"🔴 Красные: {red_count} | 🟢 Зелёные: {green_count}\n"
        
        forecast_avg = np.mean(arima_forecast)
        if forecast_avg > current_price * 1.01:
            msg += "\n🎯 <b>ПОКУПКА</b> 📈"
        elif forecast_avg < current_price * 0.99:
            msg += "\n🎯 <b>ПРОДАЖА</b> 📉"
        else:
            msg += "\n⏳ <b>ОЖИДАНИЕ</b>"
        
        if send_telegram_message(msg):
            st.session_state.messages_sent.append(moscow_time)
            return True
        return False
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return False

# MAIN
st.markdown("---")

moscow_time = get_moscow_time()

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
        if run_analysis(symbol, interval, forecast_steps):
            st.success("✅ Отчёт отправлен в Telegram!")
        else:
            st.error("❌ Ошибка отправки")

st.markdown("---")
st.subheader("📊 Данные")

with st.spinner("⏳ Загружаю данные..."):
    df = generate_demo_data(symbol)
    
    if df is not None and len(df) > 0:
        st.write("**Последние 10 свечей:**")
        display_df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        display_df = display_df.reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("📤 История отправок")
if st.session_state.messages_sent:
    data = [{"Время (МСК)": t.strftime('%Y-%m-%d %H:%M:%S')} for t in st.session_state.messages_sent[-10:]]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

st.markdown("""<script>setTimeout(() => window.location.reload(), 60000);</script>""", unsafe_allow_html=True)
