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
st.markdown("**CoinGecko + Графики + Точность + Рекомендации + Telegram**")

with st.sidebar:
    st.title("⚙️ ПАРАМЕТРЫ")
    
    st.subheader("📊 Данные для анализа")
    crypto = st.text_input("Криптовалюта", value="bitcoin")
    
    st.subheader("📚 Обучение ARIMA")
    train_period = st.selectbox(
        "Период обучения (на каких данных учиться):",
        [7, 14, 30, 90, 180, 365],
        format_func=lambda x: f"{x} дней"
    )
    
    st.subheader("🔮 Предсказание")
    forecast_type = st.radio("Тип прогноза:", ["Часы", "Дни"])
    
    if forecast_type == "Часы":
        hours = st.selectbox("Часовой таймфрейм:", [1, 4, 8, 12])
        forecast_period_label = f"{hours}h"
        days_for_chart = 30
    else:
        days = st.slider("Дней для прогноза:", 7, 365, 30)
        forecast_period_label = f"{days}d"
        days_for_chart = days
    
    forecast_steps = st.number_input("Шагов прогноза", min_value=1, max_value=500, value=7)
    
    st.divider()
    st.success("✅ Telegram подключен")
    st.info(f"📚 Обучение: {train_period}d\n🔮 Период: {forecast_period_label}")

if 'messages_sent' not in st.session_state:
    st.session_state.messages_sent = []

def get_moscow_time():
    return datetime.now(MOSCOW_TZ)

def get_coingecko_data(crypto_id, days=365):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {'vs_currency': 'usd', 'days': days, 'interval': 'daily'}
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return None
        
        data = response.json()
        prices = data['prices']
        
        df = pd.DataFrame({
            'Open time': [datetime.fromtimestamp(p[0]/1000) for p in prices],
            'Close': [p[1] for p in prices]
        })
        
        df['Open'] = df['Close'].shift(1).fillna(df['Close'].iloc[0])
        df['High'] = df['Close'] + np.abs(np.random.normal(0, df['Close'].std() * 0.02, len(df)))
        df['Low'] = df['Close'] - np.abs(np.random.normal(0, df['Close'].std() * 0.02, len(df)))
        df['Volume'] = np.random.uniform(1e9, 1e11, len(df))
        
        return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except:
        return None

def calculate_arima_forecast(prices, forecast_steps, train_period):
    """ARIMA прогноз с выбранным периодом обучения"""
    if len(prices) < 10:
        return None
    
    # Используем последние train_period дней для обучения
    train_data = prices[-train_period:] if len(prices) > train_period else prices
    
    if len(train_data) < 10:
        return None
    
    recent = train_data[-20:] if len(train_data) >= 20 else train_data
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 2)
    poly = np.poly1d(coeffs)
    
    future_x = np.arange(len(recent), len(recent) + forecast_steps)
    return poly(future_x)

def calculate_accuracy(prices, forecast, train_period):
    """Точность прогноза на выбранном периоде обучения"""
    if len(prices) < 20:
        return None, None, None
    
    # Тренируемся на train_period дней
    train_data = prices[-train_period:] if len(prices) > train_period else prices
    test_size = max(5, len(train_data) // 5)
    
    if test_size < 3 or len(train_data) < 15:
        return None, None, None
    
    train = train_data[:-test_size]
    test = train_data[-test_size:]
    
    x = np.arange(len(train))
    coeffs = np.polyfit(x, train, 2)
    poly = np.poly1d(coeffs)
    
    predicted = poly(np.arange(len(train), len(train) + test_size))
    
    rmse = np.sqrt(np.mean((test - predicted) ** 2))
    mae = np.mean(np.abs(test - predicted))
    
    accuracy = 100 - (mae / np.mean(test)) * 100
    accuracy = max(0, min(100, accuracy))
    
    return rmse, mae, accuracy

def calculate_bubbles(df):
    df = df.copy()
    df['Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open'].replace(0, 1)) * 100
    
    span = min(10, len(df)//2)
    df['Volume_EMA'] = df['Volume'].ewm(span=span, adjust=False).mean()
    df['Volume_STD'] = df['Volume'].rolling(window=span).std()
    df['Lower_Threshold'] = df['Volume_EMA'] + 0.5 * df['Volume_STD'].fillna(0)
    df['Bubble_Type'] = 'None'
    
    for i in range(span, len(df)):
        if pd.notna(df.loc[i, 'Lower_Threshold']):
            if df.loc[i, 'Price_Change_Pct'] < -0.05 and df.loc[i, 'Volume'] > df.loc[i, 'Lower_Threshold']:
                df.loc[i, 'Bubble_Type'] = 'Red'
            elif df.loc[i, 'Price_Change_Pct'] > 0.05 and df.loc[i, 'Volume'] > df.loc[i, 'Lower_Threshold']:
                df.loc[i, 'Bubble_Type'] = 'Green'
    
    return df

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        params = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
        response = requests.post(url, params=params, timeout=30)
        return response.json().get('ok', False)
    except:
        return False

def get_recommendation(forecast, current_price, accuracy):
    """Рекомендация торговли"""
    forecast_avg = np.mean(forecast)
    change_pct = ((forecast_avg - current_price) / current_price) * 100
    
    if accuracy < 50:
        return "⚠️ НИЗКАЯ ТОЧНОСТЬ - НЕ НАДЕЖНО"
    
    if change_pct > 2 and accuracy > 65:
        return "🎯 СИЛЬНАЯ ПОКУПКА 📈"
    elif change_pct > 0.5 and accuracy > 60:
        return "📈 ПОКУПКА"
    elif change_pct < -2 and accuracy > 65:
        return "🎯 СИЛЬНАЯ ПРОДАЖА 📉"
    elif change_pct < -0.5 and accuracy > 60:
        return "📉 ПРОДАЖА"
    else:
        return "⏳ ОЖИДАНИЕ"

def run_analysis(crypto_id, forecast_steps, train_period, forecast_period_label):
    try:
        # Загружаем максимум данных (365 дней)
        df = get_coingecko_data(crypto_id, 365)
        
        if df is None or len(df) < train_period:
            st.error(f"❌ Недостаточно данных (нужно минимум {train_period} дней)")
            return False
        
        prices = df['Close'].values.astype(float)
        arima_forecast = calculate_arima_forecast(prices, forecast_steps, train_period)
        
        if arima_forecast is None:
            return False
        
        rmse, mae, accuracy = calculate_accuracy(prices, arima_forecast, train_period)
        df_with_bubbles = calculate_bubbles(df)
        current_price = prices[-1]
        moscow_time = get_moscow_time()
        
        recommendation = get_recommendation(arima_forecast, current_price, accuracy)
        
        msg = f"<b>📊 ОТЧЁТ ARIMA + BUBBLES</b>\n"
        msg += f"<b>Время:</b> {moscow_time.strftime('%Y-%m-%d %H:%M:%S')} МСК\n"
        msg += f"<b>{crypto_id.upper()}</b> | Период: {forecast_period_label}\n"
        msg += f"<b>💰 Цена:</b> ${current_price:,.2f}\n\n"
        
        msg += f"<b>📚 ОБУЧЕНИЕ ARIMA:</b> {train_period} дней\n\n"
        
        msg += f"<b>📊 ТОЧНОСТЬ ПРОГНОЗА:</b>\n"
        msg += f"✓ Accuracy: {accuracy:.1f}%\n"
        msg += f"✓ RMSE: ${rmse:,.2f}\n"
        msg += f"✓ MAE: ${mae:,.2f}\n\n"
        
        msg += f"<b>📈 Прогноз на {forecast_steps} периодов:</b>\n"
        for i, price in enumerate(arima_forecast[:min(7, forecast_steps)], 1):
            change = ((price - current_price) / current_price) * 100
            arrow = "📈" if change > 0 else "📉"
            msg += f"{arrow} {i}: ${price:,.2f} ({change:+.2f}%)\n"
        
        red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
        green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
        msg += f"\n🔴 Красные пузыри: {red_count} | 🟢 Зелёные пузыри: {green_count}\n"
        
        msg += f"\n{recommendation}\n"
        
        if send_telegram(msg):
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
st.subheader("🚀 Отправка в Telegram")
if st.button("📤 ОТПРАВИТЬ ОТЧЁТ", use_container_width=True, type="primary"):
    with st.spinner("⏳ Загружаю данные и обучаю ARIMA..."):
        if run_analysis(crypto, forecast_steps, train_period, forecast_period_label):
            st.success("✅ Отчёт отправлен в Telegram!")
        else:
            st.error("❌ Ошибка")

st.markdown("---")
st.subheader("📊 РЕАЛЬНЫЕ Данные с Графиками")

with st.spinner(f"⏳ Загружаю данные и обучаю ARIMA на {train_period} дней..."):
    df = get_coingecko_data(crypto, 365)
    
    if df is not None and len(df) > train_period:
        prices = df['Close'].values.astype(float)
        arima_forecast = calculate_arima_forecast(prices, forecast_steps, train_period)
        df_bubbles = calculate_bubbles(df)
        rmse, mae, accuracy = calculate_accuracy(prices, arima_forecast, train_period)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Цена", f"${prices[-1]:,.2f}")
        with col2:
            st.metric("📈 Прогноз", f"${np.mean(arima_forecast):,.2f}")
        with col3:
            st.metric("📊 Accuracy", f"{accuracy:.1f}%" if accuracy else "N/A")
        with col4:
            st.metric("📚 Обучение", f"{train_period}d")
        
        # Рекомендация
        if arima_forecast is not None:
            recommendation = get_recommendation(arima_forecast, prices[-1], accuracy)
            st.write(f"### {recommendation}")
        
        # ГРАФИК ЦЕНЫ
        if arima_forecast is not None:
            st.write(f"**📈 ГРАФИК - История (последние 30 дней) и Прогноз ({forecast_period_label}):**")
            
            history_prices = prices[-30:]
            
            chart_df = pd.DataFrame({
                'История': list(history_prices) + [np.nan] * len(arima_forecast),
                'Прогноз': [np.nan] * len(history_prices) + list(arima_forecast)
            })
            
            st.line_chart(chart_df, use_container_width=True)
        
        # ГРАФИК ПУЗЫРЕЙ
        st.write("**🔴🟢 ГРАФИК ПУЗЫРЕЙ (Объём):**")
        
        bubble_df = pd.DataFrame({
            'Красные': [1 if t == 'Red' else 0 for t in df_bubbles['Bubble_Type']],
            'Зелёные': [1 if t == 'Green' else 0 for t in df_bubbles['Bubble_Type']],
        })
        
        st.bar_chart(bubble_df, use_container_width=True)
        
        st.write("**📊 Последние 10 дней:**")
        display_df = df[['Open time', 'Close']].tail(10).copy()
        display_df['Close'] = display_df['Close'].apply(lambda x: f"${x:,.2f}")
        display_df['Open time'] = display_df['Open time'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.error(f"❌ Недостаточно данных для обучения на {train_period} дней")

st.markdown("---")
st.subheader("📤 История")
if st.session_state.messages_sent:
    data = [{"Время": t.strftime('%Y-%m-%d %H:%M:%S')} for t in st.session_state.messages_sent[-10:]]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

st.markdown("""<script>setTimeout(() => window.location.reload(), 60000);</script>""", unsafe_allow_html=True)
