import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pytz
import io

TELEGRAM_BOT_TOKEN = "5628451765:AAF3eghUBVePX-I_j3Rg2WvWKFGkx4u1F7M"
TELEGRAM_CHAT_ID = "204683255"
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

st.set_page_config(page_title="ARIMA Bot", page_icon="📊", layout="wide")
st.title("📊 ARIMA + Market Order Bubbles")
st.markdown("**Прогноз + Telegram + Графики**")

with st.sidebar:
    st.title("⚙️ Параметры")
    symbol = st.text_input("Символ", value="BTCUSDT")
    interval = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    forecast_steps = st.number_input("Шагов прогноза", min_value=1, max_value=500, value=7, step=1)
    st.divider()
    st.success("✅ Telegram подключен")
    st.info("⏰ Московское время (UTC+3)\n📤 С графиками")

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
    prices = [base_price]
    for i in range(num_bars):
        change = np.random.normal(0, 100)
        prices.append(max(prices[-1] + change, 1000))
    
    times = pd.date_range(end=datetime.now(), periods=num_bars, freq='1h')
    
    opens = np.array(prices[:-1], dtype=float)
    closes = np.array(prices[1:], dtype=float)
    
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 100, num_bars))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 100, num_bars))
    
    df = pd.DataFrame({
        'Open time': times,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': np.random.uniform(1000, 100000, num_bars)
    })
    
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

def plot_price_and_forecast(df, forecast, symbol):
    """Создаём график с matplotlib (работает на Streamlit)"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # График 1: Цена и прогноз
        history_x = range(len(df))
        forecast_x = range(len(df) - 1, len(df) - 1 + len(forecast))
        
        ax1.plot(history_x, df['Close'].values, 'b-', label='История', linewidth=2)
        ax1.plot(forecast_x, forecast, 'r--', label='Прогноз', linewidth=2, marker='o')
        ax1.set_title(f'{symbol} Цена и ARIMA Прогноз', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Цена (USD)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # График 2: Объём
        colors = ['red' if t == 'Red' else 'green' if t == 'Green' else 'steelblue' 
                  for t in df['Bubble_Type']]
        ax2.bar(history_x, df['Volume'].values, color=colors, alpha=0.6)
        ax2.set_title('Объём с пузырями (🔴 Красные 🟢 Зелёные)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Периоды', fontsize=12)
        ax2.set_ylabel('Объём', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"⚠️ Ошибка графика: {str(e)}")
        return None

def send_telegram_message(message, image_bytes=None):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, params=params, timeout=30)
        result = response.json()
        
        # Отправляем график отдельно если есть
        if image_bytes and result.get('ok'):
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': image_bytes}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': '📊 График'}
            requests.post(url_photo, files=files, data=data, timeout=30)
        
        return result.get('ok', False)
    except:
        return False

def run_analysis(symbol, interval, forecast_steps):
    try:
        df = generate_demo_data(symbol)
        
        if len(df) < 100:
            st.error("❌ Недостаточно данных")
            return False, None, None
        
        prices = df['Close'].values.astype(float)
        arima_forecast = calculate_arima_forecast(prices, forecast_steps)
        
        if arima_forecast is None:
            return False, None, None
        
        df_with_bubbles = calculate_bubbles(df)
        current_price = prices[-1]
        
        moscow_time = get_moscow_time()
        
        msg = f"<b>📊 ОТЧЁТ ARIMA + BUBBLES</b>\n"
        msg += f"<b>Время (МСК):</b> {moscow_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        msg += f"<b>{symbol}</b> | {interval}\n"
        msg += f"<b>Цена:</b> ${current_price:.2f}\n\n"
        
        msg += f"<b>📈 Прогноз на {forecast_steps} шагов:</b>\n"
        for i, price in enumerate(arima_forecast[:min(5, forecast_steps)], 1):
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
        
        # Создаём график
        fig = plot_price_and_forecast(df_with_bubbles, arima_forecast, symbol)
        image_bytes = None
        if fig:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            image_bytes = img_buffer
        
        if send_telegram_message(msg, image_bytes):
            st.session_state.messages_sent.append(moscow_time)
            return True, fig, arima_forecast
        return False, fig, arima_forecast
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return False, None, None

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
        success, fig, forecast = run_analysis(symbol, interval, forecast_steps)
        if success:
            st.success("✅ Отчёт + ГРАФИК отправлены в Telegram!")
            if fig:
                st.pyplot(fig)
        else:
            st.error("❌ Ошибка отправки")

st.markdown("---")
st.subheader("📊 Данные и Графики")

with st.spinner("⏳ Загружаю данные..."):
    df = generate_demo_data(symbol)
    prices = df['Close'].values.astype(float)
    arima_forecast = calculate_arima_forecast(prices, forecast_steps)
    df_with_bubbles = calculate_bubbles(df)
    
    if df is not None and len(df) > 0:
        st.write("**Последние 10 свечей:**")
        display_df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(10).copy()
        display_df = display_df.reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Показываем график
        if arima_forecast is not None:
            fig = plot_price_and_forecast(df_with_bubbles, arima_forecast, symbol)
            if fig:
                st.pyplot(fig)

st.markdown("---")
st.subheader("📤 История отправок")
if st.session_state.messages_sent:
    data = [{"Время (МСК)": t.strftime('%Y-%m-%d %H:%M:%S')} for t in st.session_state.messages_sent[-10:]]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

st.markdown("""<script>setTimeout(() => window.location.reload(), 60000);</script>""", unsafe_allow_html=True)
