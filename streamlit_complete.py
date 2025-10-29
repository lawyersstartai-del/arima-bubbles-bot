import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import requests
import time
import pytz
import plotly.graph_objects as go
import io

# ========== TELEGRAM CREDENTIALS ==========
TELEGRAM_BOT_TOKEN = "5628451765:AAF3eghUBVePX-I_j3Rg2WvWKFGkx4u1F7M"
TELEGRAM_CHAT_ID = "204683255"

# ========== MOSCOW TIMEZONE ==========
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="ARIMA + Market Order Bubbles",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 ARIMA + Market Order Bubbles (WITH GRAPHS)")
st.markdown("**Автоотправка с графиками каждый час + ручная отправка**")

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("⚙️ Параметры")
    symbol = st.text_input("Символ", value="BTCUSDT")
    interval = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    forecast_steps = st.slider("Шагов прогноза", 3, 14, 7)
    days_history = st.slider("Дней истории", 7, 365, 30)
    
    st.divider()
    st.success("✅ Telegram подключен")
    st.info("⏰ Часовой пояс: Москва (UTC+3)\n📤 Автоотправка: XX:02\n📊 Графики: Plotly")
    
    st.divider()
    st.title("🤖 Автоматизация")
    auto_mode = st.checkbox("🔄 Включить автоматизацию", value=True)

# ========== STATE ==========
if 'last_send_hour' not in st.session_state:
    st.session_state.last_send_hour = -1
if 'messages_sent' not in st.session_state:
    st.session_state.messages_sent = []

# ========== HELPERS ==========
def get_moscow_time():
    return datetime.now(MOSCOW_TZ)

def get_historical_klines(symbol, interval, days):
    try:
        from binance.spot import Spot as Client
        client = Client()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        
        df = pd.DataFrame()
        limit = 1000
        
        while True:
            klines = client.klines(symbol=symbol, interval=interval, startTime=start_ts, limit=limit)
            if not klines:
                break
            
            temp_df = pd.DataFrame(klines, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            df = pd.concat([df, temp_df], ignore_index=True)
            
            last_open_time = int(temp_df['Open time'].iloc[-1])
            if len(klines) < limit:
                break
            start_ts = last_open_time + 1
        
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        
        df.drop_duplicates(subset=['Open time'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Ошибка: {str(e)}")
        return None

def calculate_arima_forecast(prices, forecast_steps=7):
    try:
        model = ARIMA(prices, order=(6, 1, 12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        
        if isinstance(forecast, pd.Series):
            return np.array(forecast.values, dtype=float)
        else:
            return np.array(forecast, dtype=float)
    except Exception as e:
        return None

def calculate_bubbles(df, stdev_length=30, ema_length=30):
    df = df.copy()
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open'].replace(0, 1)) * 100
    
    df['Volume_EMA'] = df['Volume'].ewm(span=ema_length, adjust=False).mean()
    df['Volume_STD'] = df['Volume'].rolling(window=stdev_length).std()
    
    df['Upper_Threshold'] = df['Volume_EMA'] + 1.0 * df['Volume_STD']
    df['Lower_Threshold'] = df['Volume_EMA'] + 0.5 * df['Volume_STD']
    
    df['Bubble_Type'] = 'None'
    df['Bubble_Strength'] = 0.0
    
    for i in range(max(stdev_length, ema_length), len(df)):
        current_volume = df.loc[i, 'Volume']
        current_price_change = df.loc[i, 'Price_Change_Pct']
        upper_threshold = df.loc[i, 'Upper_Threshold']
        lower_threshold = df.loc[i, 'Lower_Threshold']
        
        if pd.notna(upper_threshold) and pd.notna(lower_threshold):
            if current_price_change < -0.05 and current_volume > lower_threshold:
                strength = min(100.0, ((current_volume - lower_threshold) / (upper_threshold - lower_threshold + 0.1)) * 100)
                df.loc[i, 'Bubble_Type'] = 'Red'
                df.loc[i, 'Bubble_Strength'] = min(100.0, strength)
            
            elif current_price_change > 0.05 and current_volume > lower_threshold:
                strength = min(100.0, ((current_volume - lower_threshold) / (upper_threshold - lower_threshold + 0.1)) * 100)
                df.loc[i, 'Bubble_Type'] = 'Green'
                df.loc[i, 'Bubble_Strength'] = min(100.0, strength)
    
    return df

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=data, timeout=30)
        return response.json().get('ok', False)
    except:
        return False

def generate_recommendations(prices, arima_forecast, df_with_bubbles):
    recommendations = ""
    
    current_price = prices[-1]
    forecast_avg = np.mean(arima_forecast)
    
    if forecast_avg > current_price * 1.01:
        recommendations += "📈 <b>ARIMA:</b> РОСТ\n"
    elif forecast_avg < current_price * 0.99:
        recommendations += "📉 <b>ARIMA:</b> ПАДЕНИЕ\n"
    else:
        recommendations += "➡️ <b>ARIMA:</b> Боковик\n"
    
    red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
    green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
    
    recommendations += "\n<b>🎯 РЕКОМЕНДАЦИЯ:</b>\n"
    
    if forecast_avg > current_price and red_count > green_count:
        recommendations += "🎯 <b>ПОКУПКА</b>\n"
    elif forecast_avg < current_price and green_count > red_count:
        recommendations += "🎯 <b>ПРОДАЖА</b>\n"
    else:
        recommendations += "⏳ <b>ОЖИДАНИЕ</b>\n"
    
    return recommendations

def create_candlestick_chart(df_last):
    fig = go.Figure(data=[go.Candlestick(
        x=df_last['Open time'],
        open=df_last['Open'],
        high=df_last['High'],
        low=df_last['Low'],
        close=df_last['Close'],
        name='OHLC'
    )])
    
    fig.update_layout(
        title=f"Свечи {symbol}",
        yaxis_title="Цена (USD)",
        xaxis_title="Время",
        template="plotly_dark",
        height=400
    )
    return fig

def create_volume_chart(df_last, df_with_bubbles):
    colors = []
    for idx in df_with_bubbles.tail(len(df_last)).index:
        bubble_type = df_with_bubbles.loc[idx, 'Bubble_Type']
        if bubble_type == 'Red':
            colors.append('red')
        elif bubble_type == 'Green':
            colors.append('green')
        else:
            colors.append('steelblue')
    
    fig = go.Figure(data=[go.Bar(
        x=df_last['Open time'],
        y=df_last['Volume'],
        marker=dict(color=colors),
        name='Volume'
    )])
    
    fig.update_layout(
        title="Объём с пузырями",
        yaxis_title="Объём",
        xaxis_title="Время",
        template="plotly_dark",
        height=300
    )
    return fig

def create_forecast_chart(prices, arima_forecast):
    history = prices[-100:]
    forecast_data = arima_forecast
    
    x_hist = list(range(len(history)))
    x_fore = list(range(len(history), len(history) + len(forecast_data)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_hist,
        y=history,
        mode='lines+markers',
        name='История',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_fore,
        y=forecast_data,
        mode='lines+markers',
        name='Прогноз ARIMA',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Прогноз ARIMA(6,1,12)",
        yaxis_title="Цена (USD)",
        xaxis_title="Периоды",
        template="plotly_dark",
        height=400
    )
    return fig

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
    
    # Заголовок
    msg = f"<b>📊 ОТЧЁТ ARIMA + BUBBLES</b>\n"
    msg += f"<b>Время (МСК):</b> {moscow_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    msg += f"<b>{symbol}</b> | {interval}\n"
    msg += f"<b>Цена:</b> ${current_price:.2f}\n\n"
    
    # Прогноз
    msg += f"<b>📈 Прогноз ARIMA на {forecast_steps} шагов:</b>\n"
    for i, price in enumerate(arima_forecast, 1):
        change = ((price - current_price) / current_price) * 100
        arrow = "📈" if change > 0 else "📉"
        msg += f"{arrow} {i}: ${price:.2f} ({change:+.2f}%)\n"
    
    msg += "\n"
    
    # Пузыри
    red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
    green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
    
    msg += f"<b>🔴🟢 ПУЗЫРИ:</b>\n"
    msg += f"🔴 Красные: {red_count}\n"
    msg += f"🟢 Зелёные: {green_count}\n\n"
    
    # Рекомендация
    msg += generate_recommendations(prices, arima_forecast, df_with_bubbles)
    
    send_telegram_message(msg)
    return True

# ========== MAIN ==========
if auto_mode:
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
    st.subheader("📊 Графики")
    
    with st.spinner("⏳ Загружаю данные..."):
        df = get_historical_klines(symbol, interval, days_history)
        
        if df is not None and len(df) > 0:
            prices = df['Close'].values
            arima_forecast = calculate_arima_forecast(prices, forecast_steps)
            df_with_bubbles = calculate_bubbles(df)
            
            # Свечи
            st.plotly_chart(create_candlestick_chart(df.tail(100)), use_container_width=True)
            
            # Объём
            st.plotly_chart(create_volume_chart(df.tail(100), df_with_bubbles), use_container_width=True)
            
            # Прогноз
            if arima_forecast is not None:
                st.plotly_chart(create_forecast_chart(prices, arima_forecast), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📤 История отправок")
    if st.session_state.messages_sent:
        data = [{"Время (МСК)": t.strftime('%Y-%m-%d %H:%M:%S')} for t in st.session_state.messages_sent[-10:]]
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

st.markdown("""
<script>
setTimeout(() => window.location.reload(), 60000);
</script>
""", unsafe_allow_html=True)

st.divider()
st.markdown("<div style='text-align:center;color:gray;font-size:11px;'>🤖 ARIMA Bubbles | С графиками Plotly | Московское время (UTC+3)</div>", unsafe_allow_html=True)
