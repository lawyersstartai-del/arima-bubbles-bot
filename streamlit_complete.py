import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import matplotlib.pyplot as plt
import io
import pytz

try:
    from binance import Client
except:
    from binance.spot import Spot as Client

# ========== TELEGRAM CREDENTIALS ==========
TELEGRAM_BOT_TOKEN = "5628451765:AAF3eghUBVePX-I_j3Rg2WvWKFGkx4u1F7M"
TELEGRAM_CHAT_ID = "204683255"

# ========== MOSCOW TIMEZONE ==========
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="ARIMA + Market Order Bubbles AUTO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 ARIMA + Market Order Bubbles (MOSCOW TIME)")
st.markdown("**Автоотправка каждый час на 2-й минуте + ручная отправка по кнопке**")

# ========== SIDEBAR CONFIG ==========
with st.sidebar:
    st.title("⚙️ Параметры")
    symbol = st.text_input("Символ", value="BTCUSDT")
    interval = st.selectbox("Таймфрейм", ["1h", "4h", "1d"])
    forecast_steps = st.slider("Шагов прогноза", 3, 14, 7)
    days_history = st.slider("Дней истории", 7, 365, 30)
    
    st.divider()
    st.success("✅ Telegram подключен")
    st.info("⏰ Часовой пояс: Москва (UTC+3)\n📤 Автоотправка: XX:02 каждого часа\n🚀 Ручная отправка: по кнопке")
    
    st.divider()
    st.title("🤖 Автоматизация")
    auto_mode = st.checkbox("🔄 Включить автоматизацию", value=True)

# ========== STATE MANAGEMENT ==========
if 'last_send_hour' not in st.session_state:
    st.session_state.last_send_hour = -1
if 'messages_sent' not in st.session_state:
    st.session_state.messages_sent = []

# ========== HELPER FUNCTIONS ==========
def get_moscow_time():
    """Получает текущее время в Москве"""
    return datetime.now(MOSCOW_TZ)

def to_milliseconds(dt):
    return int(dt.timestamp() * 1000)

def get_historical_klines(symbol, interval, days):
    """Загружает исторические данные с Binance"""
    try:
        client = Client()
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        start_ts = to_milliseconds(datetime.strptime(start_date, "%Y-%m-%d"))
        
        df = pd.DataFrame()
        limit = 1000
        
        while True:
            klines = client.get_klines(symbol=symbol, interval=interval, startTime=start_ts, limit=limit)
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
        return None

def calculate_market_order_bubbles(df, stdev_length=30, ema_length=30):
    """Вычисляет Market Order Bubbles"""
    df = df.copy()
    df['Price_Change'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = (df['Price_Change'] / df['Open'].replace(0, 1)) * 100
    
    df['Volume_EMA'] = df['Volume'].ewm(span=ema_length, adjust=False).mean()
    df['Volume_STD'] = df['Volume'].rolling(window=stdev_length).std()
    
    df['Upper_Threshold'] = df['Volume_EMA'] + 1.0 * df['Volume_STD']
    df['Lower_Threshold'] = df['Volume_EMA'] + 0.5 * df['Volume_STD']
    
    df['Bubble_Type'] = 'None'
    df['Bubble_Size'] = 0
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
                
                if current_volume > upper_threshold + (upper_threshold - lower_threshold) * 0.5:
                    df.loc[i, 'Bubble_Size'] = 4.5
                elif current_volume > upper_threshold:
                    df.loc[i, 'Bubble_Size'] = 3
                else:
                    df.loc[i, 'Bubble_Size'] = 2
                
                df.loc[i, 'Bubble_Strength'] = min(100.0, strength)
            
            elif current_price_change > 0.05 and current_volume > lower_threshold:
                strength = min(100.0, ((current_volume - lower_threshold) / (upper_threshold - lower_threshold + 0.1)) * 100)
                df.loc[i, 'Bubble_Type'] = 'Green'
                
                if current_volume > upper_threshold + (upper_threshold - lower_threshold) * 0.5:
                    df.loc[i, 'Bubble_Size'] = 4.5
                elif current_volume > upper_threshold:
                    df.loc[i, 'Bubble_Size'] = 3
                else:
                    df.loc[i, 'Bubble_Size'] = 2
                
                df.loc[i, 'Bubble_Strength'] = min(100.0, strength)
    
    return df

def calculate_arima_forecast(prices, forecast_steps=7):
    """Рассчитывает прогноз ARIMA"""
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

def create_chart_with_bubbles(df_with_bubbles, prices, title=""):
    """Создаёт график свечей с пузырями"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    plot_df = df_with_bubbles.iloc[-100:].copy()
    plot_df.reset_index(drop=True, inplace=True)
    
    for idx, row in plot_df.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        ax.bar(idx, row['Close'] - row['Open'], bottom=row['Open'], color=color, width=0.6, alpha=0.7)
        ax.plot([idx, idx], [row['Low'], row['High']], color='black', linewidth=0.5)
    
    for idx, row in plot_df.iterrows():
        if row['Bubble_Type'] == 'Red':
            ax.scatter(idx, row['Low'], color='red', s=row['Bubble_Size']*200, marker='v', zorder=5, edgecolors='darkred', linewidth=2)
        elif row['Bubble_Type'] == 'Green':
            ax.scatter(idx, row['High'], color='lime', s=row['Bubble_Size']*200, marker='^', zorder=5, edgecolors='darkgreen', linewidth=2)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_xlabel("Свечи", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_volume_chart(df_with_bubbles):
    """Создаёт график объёма"""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    plot_df = df_with_bubbles.iloc[-100:].copy()
    plot_df.reset_index(drop=True, inplace=True)
    
    volumes = plot_df['Volume'].values
    bubble_types = plot_df['Bubble_Type'].values
    colors = ['red' if bt == 'Red' else ('lime' if bt == 'Green' else 'steelblue') for bt in bubble_types]
    
    ax.bar(range(len(volumes)), volumes, color=colors, alpha=0.7)
    ax.set_title("Объём с Market Order Bubbles", fontsize=14, fontweight='bold')
    ax.set_ylabel("Объём", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def create_forecast_chart(prices, arima_forecast):
    """Создаёт график прогноза"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    history = prices[-100:]
    forecast_data = arima_forecast
    
    x_hist = range(len(history))
    x_fore = range(len(history), len(history) + len(forecast_data))
    
    ax.plot(x_hist, history, label='История', color='blue', linewidth=2.5, marker='o', markersize=3)
    ax.plot(x_fore, forecast_data, label='Прогноз ARIMA', color='red', linewidth=2.5, marker='s', linestyle='--', markersize=5)
    ax.axvline(x=len(history)-1, color='gray', linestyle=':', linewidth=2)
    ax.fill_between(x_fore, forecast_data*0.99, forecast_data*1.01, alpha=0.2, color='red')
    
    ax.set_title("Прогноз ARIMA(6,1,12)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Цена (USD)", fontsize=12)
    ax.set_xlabel("Временные периоды", fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def fig_to_png_bytes(fig):
    """Конвертирует figure в PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def send_telegram_photo(photo_bytes, caption):
    """Отправляет фото в Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {'photo': photo_bytes}
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': caption,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        return response.json().get('ok', False)
    except:
        return False

def send_telegram_message(message):
    """Отправляет текст в Telegram"""
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
    """Генерирует рекомендации"""
    recommendations = ""
    
    current_price = prices[-1]
    forecast_avg = np.mean(arima_forecast)
    
    if forecast_avg > current_price * 1.01:
        recommendations += "📈 <b>ARIMA:</b> РОСТ\n"
    elif forecast_avg < current_price * 0.99:
        recommendations += "📉 <b>ARIMA:</b> ПАДЕНИЕ\n"
    else:
        recommendations += "➡️ <b>ARIMA:</b> Боковик\n"
    
    recent_bubbles = df_with_bubbles[df_with_bubbles['Bubble_Type'] != 'None'].tail(3)
    
    if len(recent_bubbles) > 0:
        last_bubble = recent_bubbles.iloc[-1]
        
        if last_bubble['Bubble_Type'] == 'Red':
            recommendations += "🔴 <b>ПУЗЫРЬ:</b> ПРОДАЖИ → ОТСКОК вверх\n"
        else:
            recommendations += "🟢 <b>ПУЗЫРЬ:</b> ПОКУПКИ → ОТКАТ вниз\n"
    
    recommendations += "\n<b>🎯 РЕКОМЕНДАЦИЯ:</b>\n"
    
    red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
    green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
    
    if forecast_avg > current_price and red_count > green_count:
        recommendations += "🎯 <b>ПОКУПКА</b>\n"
    elif forecast_avg < current_price and green_count > red_count:
        recommendations += "🎯 <b>ПРОДАЖА</b>\n"
    else:
        recommendations += "⏳ <b>ОЖИДАНИЕ</b>\n"
    
    recommendations += f"\n📊 Статистика:\n"
    recommendations += f"   🔴 Красные: {red_count} | 🟢 Зелёные: {green_count}\n"
    
    return recommendations

def run_hourly_analysis(symbol, interval, forecast_steps, days_history):
    """Выполняет полный анализ"""
    
    df = get_historical_klines(symbol, interval, days_history)
    
    if df is None or len(df) < 100:
        return False
    
    prices = df['Close'].values
    arima_forecast = calculate_arima_forecast(prices, forecast_steps)
    
    if arima_forecast is None:
        return False
    
    df_with_bubbles = calculate_market_order_bubbles(df)
    current_price = prices[-1]
    
    moscow_time = get_moscow_time()
    
    # Заголовок
    header_msg = f"<b>📊 АВТОМАТИЧЕСКИЙ ОТЧЁТ</b>\n"
    header_msg += f"<b>Время (МСК):</b> {moscow_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    header_msg += f"<b>{symbol}</b> | {interval}\n"
    header_msg += f"<b>Цена:</b> ${current_price:.2f}\n"
    send_telegram_message(header_msg)
    time.sleep(0.5)
    
    # Свечи
    fig = create_chart_with_bubbles(df_with_bubbles, prices, f"Свечи {symbol}")
    send_telegram_photo(fig_to_png_bytes(fig), "📊 График свечей с пузырями")
    time.sleep(0.5)
    
    # Объём
    fig = create_volume_chart(df_with_bubbles)
    send_telegram_photo(fig_to_png_bytes(fig), "📊 График объёма")
    time.sleep(0.5)
    
    # Прогноз
    fig = create_forecast_chart(prices, arima_forecast)
    send_telegram_photo(fig_to_png_bytes(fig), "📈 Прогноз ARIMA")
    time.sleep(0.5)
    
    # Таблица
    forecast_msg = f"<b>Прогноз на {forecast_steps} шагов:</b>\n\n"
    for i, price in enumerate(arima_forecast, 1):
        change = ((price - current_price) / current_price) * 100
        arrow = "📈" if change > 0 else "📉"
        forecast_msg += f"{arrow} {i}: ${price:.2f} ({change:+.2f}%)\n"
    send_telegram_message(forecast_msg)
    time.sleep(0.5)
    
    # Пузыри
    red_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Red'])
    green_count = len(df_with_bubbles[df_with_bubbles['Bubble_Type'] == 'Green'])
    
    bubbles_msg = f"<b>🔴🟢 ПУЗЫРИ</b>\n"
    bubbles_msg += f"🔴 {red_count} | 🟢 {green_count}\n\n"
    
    recent = df_with_bubbles[df_with_bubbles['Bubble_Type'] != 'None'].tail(3)
    if len(recent) > 0:
        bubbles_msg += "<b>Последние:</b>\n"
        for idx, row in recent.iterrows():
            size = 'S' if row['Bubble_Size'] == 2 else ('M' if row['Bubble_Size'] == 3 else 'L')
            t = "🔴" if row['Bubble_Type'] == 'Red' else "🟢"
            bubbles_msg += f"{t} {size} {row['Bubble_Strength']:.0f}%\n"
    
    send_telegram_message(bubbles_msg)
    time.sleep(0.5)
    
    # Рекомендация
    send_telegram_message(generate_recommendations(prices, arima_forecast, df_with_bubbles))
    
    return True

# ========== MAIN ==========
if auto_mode:
    st.markdown("---")
    
    moscow_time = get_moscow_time()
    current_hour = moscow_time.hour
    current_minute = moscow_time.minute
    
    # ✅ ОТПРАВЛЯЕМ НА 2-й МИНУТЕ КАЖДОГО ЧАСА (2:02, 3:02, 4:02 и т.д.)
    should_send = (current_minute == 2) and (st.session_state.last_send_hour != current_hour)
    
    if should_send:
        with st.spinner("⏳ Отправляю отчёт в Telegram (МСК)..."):
            if run_hourly_analysis(symbol, interval, forecast_steps, days_history):
                st.session_state.last_send_hour = current_hour
                st.session_state.messages_sent.append(moscow_time)
                st.success(f"✅ Отчёт отправлен в {moscow_time.strftime('%H:%M:%S')} МСК")
    
    # Информация
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🕐 Время (МСК)", moscow_time.strftime('%H:%M:%S'))
    with col2:
        st.metric("📤 Отправлено", len(st.session_state.messages_sent))
    with col3:
        st.metric("🤖 Статус", "🟢 РАБОТАЕТ")
    
    st.markdown("---")
    st.subheader("🚀 Ручная отправка")
    if st.button("📤 ОТПРАВИТЬ ОТЧЁТ ПРЯМО СЕЙЧАС", use_container_width=True, type="primary"):
        with st.spinner("⏳ Отправляю в Telegram..."):
            if run_hourly_analysis(symbol, interval, forecast_steps, days_history):
                st.session_state.messages_sent.append(get_moscow_time())
                st.success("✅ Отчёт отправлен!")
            else:
                st.error("❌ Ошибка отправки")
    
    # История
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
st.markdown("<div style='text-align:center;color:gray;font-size:11px;'>🤖 ARIMA + Bubbles AUTO v4.0 | Московское время (UTC+3) | Автоотправка XX:02 | Ручная отправка по кнопке</div>", unsafe_allow_html=True)
