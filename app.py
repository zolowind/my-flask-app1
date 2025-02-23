from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ Force Matplotlib into non-GUI mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime

app = Flask(__name__)

def calculate_rsi(series, period=14):
    delta = series.diff().squeeze()
    gain = np.where(delta > 0, delta, 0).squeeze()
    loss = np.where(delta < 0, -delta, 0).squeeze()

    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)

    avg_gain = gain_series.ewm(span=period, adjust=False).mean().squeeze()
    avg_loss = loss_series.ewm(span=period, adjust=False).mean().squeeze()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return pd.Series(rsi, index=series.index).fillna(50)

def calculate_macd(data, short=12, long=26, signal=9):
    short_ema = data['Close'].ewm(span=short, adjust=False).mean().squeeze()
    long_ema = data['Close'].ewm(span=long, adjust=False).mean().squeeze()
    macd = (short_ema - long_ema).squeeze()
    signal_line = macd.ewm(span=signal, adjust=False).mean().squeeze()
    return macd, signal_line

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        try:
            end_date = datetime.today().strftime("%Y-%m-%d")  # ✅ Current Date
            data = yf.download(stock_symbol, start="2020-01-01", end=end_date, progress=False)

            if data.empty:
                return render_template('index.html', error="No data found for the given stock symbol.")

            data['SMA50'] = data['Close'].rolling(window=50).mean().squeeze()
            data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean().squeeze()
            data['RSI14'] = calculate_rsi(data['Close'], period=14)
            data['MACD'], data['MACD_signal'] = calculate_macd(data)

            data['Buy_Signal'] = (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))
            data['Sell_Signal'] = (data['RSI14'] > 70) & (data['RSI14'].shift(1) <= 70)

            # ✅ Check Buy/Sell Signal for the Current Date
            today_signal = "No Clear Signal"
            if not data.empty:
                latest_macd = data['MACD'].iloc[-1]
                latest_signal = data['MACD_signal'].iloc[-1]
                latest_rsi = data['RSI14'].iloc[-1]

                if latest_macd > latest_signal and latest_rsi < 70:
                    today_signal = "BUY"
                elif latest_rsi > 70:
                    today_signal = "SELL"

            charts = {}
            for chart_type in ['price', 'rsi', 'signals']:
                fig, ax = plt.subplots(figsize=(10, 5))

                if chart_type == 'price':
                    ax.plot(data.index, data['Close'].squeeze(), label='Close Price', color='blue')
                    ax.plot(data.index, data['SMA50'].squeeze(), label='SMA 50', color='orange', linestyle='--')
                    ax.plot(data.index, data['EMA50'].squeeze(), label='EMA 50', color='green', linestyle='--')
                    ax.set_title(f'{stock_symbol} Stock Price with SMA & EMA')

                elif chart_type == 'rsi':
                    ax.plot(data.index, data['RSI14'].squeeze(), label='RSI 14', color='purple')
                    ax.axhline(70, color='red', linestyle='--', label='Overbought (70)')
                    ax.axhline(30, color='green', linestyle='--', label='Oversold (30)')
                    ax.fill_between(data.index, 30, 70, color='gray', alpha=0.2)
                    ax.set_ylim(0, 100)
                    ax.set_title(f'{stock_symbol} Relative Strength Index (RSI)')
                    ax.legend()

                elif chart_type == 'signals':
                    ax.plot(data.index, data['Close'].squeeze(), label='Close Price', color='blue')
                    ax.plot(data.index, data['SMA50'].squeeze(), label='SMA 50', color='orange', linestyle='--')
                    ax.scatter(data[data['Buy_Signal']].index, data['Close'][data['Buy_Signal']].squeeze(),
                               marker='^', s=100, color='green', label='Buy Signal', alpha=0.7)
                    ax.scatter(data[data['Sell_Signal']].index, data['Close'][data['Sell_Signal']].squeeze(),
                               marker='v', s=100, color='red', label='Sell Signal', alpha=0.7)
                    ax.set_title(f'{stock_symbol} Buy & Sell Signals')

                ax.legend()
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                charts[chart_type] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

            return render_template('index.html', stock_symbol=stock_symbol, charts=charts, today_signal=today_signal)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html', stock_symbol=None)

if __name__ == '__main__':
    app.run(debug=True)  # ✅ Flask will work without Tkinter errors
