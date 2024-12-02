import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import yfinance as yf
import plotly.graph_objs as go
from nselib import capital_market
from nselib import derivatives
from streamlit_lottie import st_lottie
import requests
from textblob import TextBlob

st.set_page_config(page_title='Stock Market Prediction ', page_icon=':chart_with_upwards_trend:')

# Function to create a database connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(e)
    return conn

# Function to create a new user
def create_user(conn, username, email, password):
    cur = conn.cursor()
    # Check if the user already exists
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    existing_user = cur.fetchone()
    if existing_user:
        return None, "User with this email already exists. Please use a different email."
    else:
        try:
            sql = ''' INSERT INTO users(username,email,password)
                      VALUES(?,?,?) '''
            cur.execute(sql, (username, email, password))
            conn.commit()
            return cur.lastrowid, None
        except sqlite3.Error as e:
            return None, str(e)

# Function to check if a user exists in the database
def check_user(conn, email, password):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = cur.fetchone()
    return user

lottie_urls = {
    "home": "https://lottie.host/0ef46eab-d90a-42e0-8709-28c337699dc1/XMqrZCglQR.json",
    "screener": "https://lottie.host/2fe89f7a-68cb-49d7-a699-e2e2d88df54b/IVVFddwe7Z.json",
    "tools": "https://lottie.host/cd46ee9c-a82b-48ae-af7d-a80b0658a7ec/jJvr9GFylg.json",
    "about": "https://lottie.host/e5797a37-65c0-44df-8d92-38f72eee4fd9/ogHow1IeLu.json",
    "logout": "https://lottie.host/1c1aa709-2ed7-4fec-b106-012053026d9f/75rl8Nm1rw.json",
    # Add more pages and their respective Lottie animation URLs here
}

# Function to display the Lottie animation for a specific page
def display_lottie_animation(page_key):
    animation_url = lottie_urls.get(page_key)
    if animation_url:
        response = requests.get(animation_url)
        animation_json = response.json()
        st_lottie(animation_json, speed=1, width=800, height=500, key=f"lottie_{page_key}_animation")

# Define functions for each page
def show_home():
    st.subheader("Market Data")
    display_lottie_animation("home")
    def fetch_equity_market_data(data_info):
            if data_info in ['equity_list', 'fno_equity_list', 'market_watch_all_indices', 'nifty50_equity_list']:
                return getattr(capital_market, data_info)()
            elif data_info in ['bhav_copy_equities', 'bhav_copy_with_delivery']:
                date = st.sidebar.text_input('Date', '22-12-23')
                parsed_date = datetime.strptime(date, '%d-%m-%y')
                formatted_date = parsed_date.strftime('%d-%m-%Y')
                return getattr(capital_market, data_info)(formatted_date)
            elif data_info in ['block_deals_data', 'bulk_deal_data', 'india_vix_data', 'short_selling_data']:
                period_ = st.sidebar.text_input('Period', '1M')
                return getattr(capital_market, data_info)(period=period_)
    def fetch_derivatives_market_data(data_info):
            if data_info in ['expiry_dates_future', 'expiry_dates_option_index']:
                return getattr(derivatives, data_info)()
            elif data_info in ['fii_derivatives_statistics', 'fno_bhav_copy', 'participant_wise_open_interest', 'participant_wise_trading_volume']:
                date = st.sidebar.text_input('Date', '22-12-23')
                parsed_date = datetime.strptime(date, '%d-%m-%y')
                formatted_date = parsed_date.strftime('%d-%m-%Y')
                return getattr(derivatives, data_info)(formatted_date)
            elif data_info == 'future_price_volume_data':
                ticker = st.sidebar.text_input('Ticker', 'SBIN')
                type_ = st.sidebar.text_input('Tnstrument Type', 'FUTSTK')
                period_ = st.sidebar.text_input('Period', '1M')
                return derivatives.future_price_volume_data(ticker, type_, period=period_)
            elif data_info == 'option_price_volume_data':
                ticker = st.sidebar.text_input('Ticker', 'BANKNIFTY')
                type_ = st.sidebar.text_input('Tnstrument Type', 'OPTIDX')
                period_ = st.sidebar.text_input('Period', '1M')
                return derivatives.option_price_volume_data(ticker, type_, period=period_)
            elif data_info == 'nse_live_option_chain':
                ticker = st.sidebar.text_input('Ticker', 'BANKNIFTY')
                expiry_data = st.sidebar.text_input('Expiry Data', '28-12-2023')
                return derivatives.nse_live_option_chain(ticker, expiry_date=expiry_data)
        
    def main():       
           st.header('Indian Stock Dashboard')
           instrument = st.radio('Instrument type', options=('NSC Equity Market', 'NSC Derivatives Market'))
           if instrument == 'NSC Equity Market':
                data_info = st.selectbox('Data to extract', options=('select the button' ,'bhav_copy_equities', 'bhav_copy_with_delivery', 
                                                                      'equity_list', 'fno_equity_list',
                                                                      'market_watch_all_indices', 'nifty50_equity_list', 
                                                                      'block_deals_data', 'bulk_deal_data',
                                                                      'india_vix_data', 'short_selling_data', 
                                                                      ))
                data = fetch_equity_market_data(data_info)

           elif instrument == 'NSC Derivatives Market':
                data_info = st.selectbox('Data to extract', options=('select the button' ,'expiry_dates_future', 'expiry_dates_option_index',
                                                                   'fno_bhav_copy', 'future_price_volume_data', 
                                                                     'nse_live_option_chain', 'option_price_volume_data', 
                                                                     'participant_wise_open_interest', 
                                                                     'participant_wise_trading_volume'))
                data = fetch_derivatives_market_data(data_info)
           st.write(data)
    if __name__ == "__main__":
        main()  
        


    if st.session_state.logged_in:
            st.write("You are logged in.")
    else:
            st.write("You are not logged in.")

def calculate_valuation_scores(info):
    # Define scoring ranges (you can adjust these as needed)
    pe_range = [(0, 15), (15, 25), (25, 35), (35, float('inf'))]  # P/E ratio
    ps_range = [(0, 1), (1, 2), (2, 3), (3, float('inf'))]  # P/S ratio
    pb_range = [(0, 1), (1, 2), (2, 3), (3, float('inf'))]  # P/B ratio
    ev_to_rev_range = [(0, 10), (10, 20), (20, 30), (30, float('inf'))]  # Enterprise Value to Revenue
    ev_to_ebitda_range = [(0, 10), (10, 15), (15, 20), (20, float('inf'))]  # Enterprise Value to EBITDA

    # Get fundamental metrics
    pe_ratio = info.get('forwardPE', 0)
    ps_ratio = info.get('priceToSalesTrailing12Months', 0)
    pb_ratio = info.get('priceToBook', 0)
    ev_to_rev = info.get('enterpriseToRevenue', 0)
    ev_to_ebitda = info.get('enterpriseToEbitda', 0)

    # Calculate scores and labels
    pe_score, pe_label = calculate_score_and_label(pe_ratio, pe_range)
    ps_score, ps_label = calculate_score_and_label(ps_ratio, ps_range)
    pb_score, pb_label = calculate_score_and_label(pb_ratio, pb_range)
    ev_to_rev_score, ev_to_rev_label = calculate_score_and_label(ev_to_rev, ev_to_rev_range)
    ev_to_ebitda_score, ev_to_ebitda_label = calculate_score_and_label(ev_to_ebitda, ev_to_ebitda_range)

    return pe_score, pe_label, ps_score, ps_label, pb_score, pb_label, ev_to_rev_score, ev_to_rev_label, ev_to_ebitda_score, ev_to_ebitda_label

# Function to calculate score and label based on value and ranges
def calculate_score_and_label(value, ranges):
    for i, (lower, upper) in enumerate(ranges):
        if lower <= value < upper:
            if i == 0:
                label = "Attractive"
            elif i == len(ranges) - 1:
                label = "Overpriced"
            else:
                label = "Expensive"
            return i + 1, label  # Scores start from 1
    return len(ranges), "Overpriced"  # If value exceeds last range, return the maximum score


def show_screener():
    display_lottie_animation("screener")
    st.title('Indian Stock Market Stocks Charts')
    symbol = st.text_input("Enter a stock symbol or index (e.g., AAPL for Apple, BANKNIFTY for BankNifty):")

    if symbol:
        try:
            if symbol.upper() == "BANKNIFTY":
                symbol = "^NSEBANK"  # Yahoo Finance symbol for BankNifty
            stock = yf.Ticker(symbol)
            info = stock.info

            st.subheader("Company Information" if symbol.upper() != "^NSEBANK" else "Index Information")
            st.write("Company Name/Index:", info.get('longName', 'N/A'))
            st.write("Sector:", info.get('sector', 'N/A'))
            st.write("Industry:", info.get('industry', 'N/A'))
            st.write("Market Cap/Index Cap:", info.get('marketCap', 'N/A'))
            st.write("Country:", info.get('country', 'N/A'))
            st.write("Website:", info.get('website', 'N/A'))
            st.write("Summary:", info.get('longBusinessSummary', 'N/A'))

            if symbol.upper() != "^NSEBANK":
                st.subheader("Fundamental Analysis")
                st.write("Forward P/E Ratio:", info.get('forwardPE', 'N/A'))
                st.write("Trailing P/E Ratio:", info.get('trailingPE', 'N/A'))
                st.write("Price to Sales Ratio (P/S):", info.get('priceToSalesTrailing12Months', 'N/A'))
                st.write("Price to Book Ratio (P/B):", info.get('priceToBook', 'N/A'))
                st.write("Enterprise Value to Revenue:", info.get('enterpriseToRevenue', 'N/A'))
                st.write("Enterprise Value to EBITDA:", info.get('enterpriseToEbitda', 'N/A'))
                st.write("Dividend Yield:", info.get('dividendYield', 'N/A'))
                st.write("Profit Margin:", info.get('profitMargins', 'N/A'))
                st.write("Beta:", info.get('beta', 'N/A'))

                # Calculate valuation scores and labels
                pe_score, pe_label, ps_score, ps_label, pb_score, pb_label, ev_to_rev_score, ev_to_rev_label, ev_to_ebitda_score, ev_to_ebitda_label = calculate_valuation_scores(info)

                st.subheader("Valuation Scores")
                st.write("Forward P/E Ratio Score:", pe_score, "(" + pe_label + ")")
                st.write("Price to Sales Ratio (P/S) Score:", ps_score, "(" + ps_label + ")")
                st.write("Price to Book Ratio (P/B) Score:", pb_score, "(" + pb_label + ")")
                st.write("Enterprise Value to Revenue Score:", ev_to_rev_score, "(" + ev_to_rev_label + ")")
                st.write("Enterprise Value to EBITDA Score:", ev_to_ebitda_score, "(" + ev_to_ebitda_label + ")")

                # Display gauge chart for valuation scores
                st.subheader("Valuation Scores Gauge Chart")
                fig = go.Figure()

                # Define gauge chart for each score
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=pe_score,
                    domain={'x': [0, 0.25], 'y': [0.5, 0.9]},
                    title={'text': "P/E Ratio Score"},
                    gauge={'axis': {'range': [None, 4]},
                           'bar': {'color': "darkblue"},
                           'steps': [
                               {'range': [0, 1], 'color': "red"},
                               {'range': [1, 2], 'color': "orange"},
                               {'range': [2, 3], 'color': "yellow"},
                               {'range': [3, 4], 'color': "green"}],
                           'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 3}},
                    number={'suffix': " (" + pe_label + ")"}
                ))

                # Repeat the above for other scores (P/S, P/B, EV/Rev,
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Current Price")
            price = info.get('regularMarketPrice', 'N/A')
            change = info.get('regularMarketChange', 'N/A')
            prev_close = info.get('regularMarketPreviousClose', 'N/A')
            open_price = info.get('regularMarketOpen', 'N/A')
            high = info.get('regularMarketDayHigh', 'N/A')
            low = info.get('regularMarketDayLow', 'N/A')
            st.write("Price:", info['currency'], price)
            st.write("Change:", info['currency'], change)
            st.write("Previous Close:", info['currency'], prev_close)
            st.write("Open:", info['currency'], open_price)
            st.write("High:", info['currency'], high)
            st.write("Low:", info['currency'], low)

            st.subheader("Historical Price Chart")
            history = stock.history(period="1y")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history.index, y=history['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title=f"{symbol} Historical Price Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Technical Analysis")
            st.write("*Note: Technical indicators are for educational purposes only and should not be considered financial advice.*")

            # Example of adding moving averages (50-day and 200-day) to the chart
            history['MA50'] = history['Close'].rolling(window=50).mean()
            history['MA200'] = history['Close'].rolling(window=200).mean()

            fig.add_trace(go.Scatter(x=history.index, y=history['MA50'], mode='lines', name='50-Day MA'))
            fig.add_trace(go.Scatter(x=history.index, y=history['MA200'], mode='lines', name='200-Day MA'))

            st.plotly_chart(fig, use_container_width=True)

            # Add an option to compare multiple stocks
            st.subheader("Compare Stocks")
            stocks_to_compare = st.text_input("Enter symbols separated by commas (e.g., AAPL,GOOGL):").upper().split(",")
            if stocks_to_compare:
                for symbol in stocks_to_compare:
                    stock_data = yf.Ticker(symbol)
                    st.write(f"**{symbol}**")
                    st.write(stock_data.history(period="1y"))
        except ValueError:
            st.error("Invalid symbol. Please enter a valid stock symbol or index.")

def fetch_stock_news(symbol):
    api_key = '478378216e84426193f454f310abadb6'
    url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        return None

def show_tools():
    st.title('Stock News Show')
    display_lottie_animation("tools")
    # Add tools page content here
    def fetch_stock_data(symbol, start_date, end_date):
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            return stock_data

    def display_stock_chart(stock_data):
              fig = go.Figure()
              fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
              fig.layout.update(title='Stock Price Chart', xaxis_title='Date', yaxis_title='Price')
              st.plotly_chart(fig)

    symbol = st.sidebar.text_input("Enter Stock Ticker Symbol", value='AAPL')
    start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
    end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2024-01-01'))

    if st.sidebar.button('Fetch Data'):
        try:
            stock_data = fetch_stock_data(symbol, start_date, end_date)
            st.success('Data fetched successfully!')
            st.write(stock_data.head())
            display_stock_chart(stock_data)
        except Exception as e:
            st.error(f'Error fetching data: {str(e)}')
    # Embedding TradingView chart with custom width and height
    st.components.v1.html(f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="width: 100vw; height: 100vh;">
      <div id="tradingview_chart" style="width: 100%; height: 100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget(
        {{
          "autosize": true,
          "symbol": "{symbol}",
          "interval": "D",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "container_id": "tradingview_chart",
          "range": "{start_date} {end_date}"
        }}
        );
      </script>
    </div>
    <!-- TradingView Widget END -->
    """, height=600)

    if st.sidebar.button("Search"):
        if symbol:
            news = fetch_stock_news(symbol)
            if news:
                st.write(f"Latest news for {symbol}:")
                for article in news:
                    st.write("Headline:", article['title'])
                    st.write("Source:", article['source']['name'])
                    st.write("Published:", article['publishedAt'])
                    st.write("URL:", article['url'])
                    st.write("---")
            else:
                st.write("Failed to fetch news. Please try again later.")

def show_technical_analysis():
    st.title('Technical Analysis')
    display_lottie_animation("technical_analysis")

    def get_stock_data(symbol, start_date, end_date, interval):
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        return stock_data
    def display_stock_data(stock_data):
        st.subheader('Stock Prices')
        fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])
        st.plotly_chart(fig, use_container_width=True)

        # Moving average
        st.subheader('Moving Average')
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='MA50'))
        fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='MA200'))
        st.plotly_chart(fig_ma, use_container_width=True)

        # Volume
        st.subheader('Volume')
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume']))
        st.plotly_chart(fig_volume, use_container_width=True)

    # Function to fetch recent news
    def get_recent_news(symbol):
        news = yf.Ticker(symbol).news
        return news

    # Function to display recent news
    def display_recent_news(news):
        st.subheader('Recent News')
        if len(news) > 0:
            for item in news:
                st.write(f"- {item['title']}")
        else:
            st.write("No recent news found.")

    # Function to display portfolio tracker
    def portfolio_tracker():
        st.sidebar.title('Portfolio Tracker')
        portfolio = st.sidebar.text_input('Enter Stock Symbol', '').upper()
        if st.sidebar.button('Add to Portfolio'):
            if 'portfolio' not in st.session_state:
                st.session_state.portfolio = []
            if portfolio not in st.session_state.portfolio:
                st.session_state.portfolio.append(portfolio)

        if 'portfolio' in st.session_state:
            st.sidebar.subheader('Portfolio')
            for item in st.session_state.portfolio:
                st.sidebar.write(item)

    # Function to display technical indicators
    def display_technical_indicators(stock_data):
        st.sidebar.title('Technical Indicators')
        if st.sidebar.checkbox('Relative Strength Index (RSI)'):
            st.subheader('Relative Strength Index (RSI)')
            delta = stock_data['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            RS = gain / loss
            RSI = 100 - (100 / (1 + RS))
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=RSI, mode='lines', name='RSI'))
            st.plotly_chart(fig_rsi, use_container_width=True)

        if st.sidebar.checkbox('Moving Average Convergence Divergence (MACD)'):
            st.subheader('Moving Average Convergence Divergence (MACD)')
            exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=stock_data.index, y=macd, mode='lines', name='MACD'))
            fig_macd.add_trace(go.Scatter(x=stock_data.index, y=signal, mode='lines', name='Signal'))
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Moving Averages
        if st.sidebar.checkbox('Moving Averages'):
            st.subheader('Moving Averages')
            stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='MA50'))
            fig_ma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='MA200'))
            st.plotly_chart(fig_ma, use_container_width=True)

        # Bollinger Bands
        if st.sidebar.checkbox('Bollinger Bands'):
            st.subheader('Bollinger Bands')
            stock_data['MA20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['20D_STD'] = stock_data['Close'].rolling(window=20).std()
            stock_data['UpperBand'] = stock_data['MA20'] + (stock_data['20D_STD'] * 2)
            stock_data['LowerBand'] = stock_data['MA20'] - (stock_data['20D_STD'] * 2)
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))
            fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['UpperBand'], mode='lines', name='Upper Band'))
            fig_bb.add_trace(go.Scatter(x=stock_data.index, y=stock_data['LowerBand'], mode='lines', name='Lower Band'))
            st.plotly_chart(fig_bb, use_container_width=True)

        # Stochastic Oscillator
        if st.sidebar.checkbox('Stochastic Oscillator'):
            st.subheader('Stochastic Oscillator')
            period = 14
            stock_data['LowestLow'] = stock_data['Low'].rolling(window=period).min()
            stock_data['HighestHigh'] = stock_data['High'].rolling(window=period).max()
            stock_data['%K'] = ((stock_data['Close'] - stock_data['LowestLow']) / (stock_data['HighestHigh'] - stock_data['LowestLow'])) * 100
            stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()
            fig_stochastic = go.Figure()
            fig_stochastic.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%K'], mode='lines', name='%K'))
            fig_stochastic.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%D'], mode='lines', name='%D'))
            st.plotly_chart(fig_stochastic, use_container_width=True)

        # Volume Oscillator
        if st.sidebar.checkbox('Volume Oscillator'):
            st.subheader('Volume Oscillator')
            short_window = 12
            long_window = 26
            stock_data['Short_MA_Volume'] = stock_data['Volume'].rolling(window=short_window).mean()
            stock_data['Long_MA_Volume'] = stock_data['Volume'].rolling(window=long_window).mean()
            stock_data['Volume_Oscillator'] = stock_data['Short_MA_Volume'] - stock_data['Long_MA_Volume']
            fig_volume_oscillator = go.Figure()
            fig_volume_oscillator.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Volume_Oscillator'], mode='lines', name='Volume Oscillator'))
            st.plotly_chart(fig_volume_oscillator, use_container_width=True)
            
        if st.sidebar.checkbox('Ichimoku Cloud'):
            st.subheader('Ichimoku Cloud')
            conversion_period = 9
            base_period = 26
            lagging_period = 52
            displacement = 26
            high_prices = stock_data['High']
            low_prices = stock_data['Low']
            tenkan_sen = (high_prices.rolling(window=conversion_period).max() + low_prices.rolling(window=conversion_period).min()) / 2
            kijun_sen = (high_prices.rolling(window=base_period).max() + low_prices.rolling(window=base_period).min()) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            senkou_span_b = ((high_prices.rolling(window=lagging_period).max() + low_prices.rolling(window=lagging_period).min()) / 2).shift(displacement)
            chikou_span = stock_data['Close'].shift(-displacement)
            
            fig_ichimoku = go.Figure()
            fig_ichimoku.add_trace(go.Scatter(x=stock_data.index, y=senkou_span_a, mode='lines', name='Senkou Span A', line=dict(color='blue')))
            fig_ichimoku.add_trace(go.Scatter(x=stock_data.index, y=senkou_span_b, mode='lines', name='Senkou Span B', line=dict(color='red')))
            fig_ichimoku.add_trace(go.Scatter(x=stock_data.index, y=tenkan_sen, mode='lines', name='Tenkan Sen', line=dict(color='green')))
            fig_ichimoku.add_trace(go.Scatter(x=stock_data.index, y=kijun_sen, mode='lines', name='Kijun Sen', line=dict(color='orange')))
            fig_ichimoku.add_trace(go.Scatter(x=stock_data.index, y=chikou_span, mode='lines', name='Chikou Span', line=dict(color='purple')))
            st.plotly_chart(fig_ichimoku, use_container_width=True)

    # Main function
    def main():
        # Sidebar for user input
        st.sidebar.title('User Input')
        symbol = st.sidebar.text_input('Enter Stock Symbol', value='AAPL', max_chars=20).upper()
        start_date = st.sidebar.date_input('Start Date', value=datetime(2020, 1, 1))
        end_date = st.sidebar.date_input('End Date', value=datetime.today())
        interval = st.sidebar.selectbox('Select Interval', ['1d', '1wk', '1mo'])

        # Fetch and display stock data
        try:
            stock_data = get_stock_data(symbol, start_date, end_date, interval)
            if not stock_data.empty:
                display_stock_data(stock_data)
            else:
                st.error("No data found for the selected symbol.")
            
            # Fetch and display recent news
            recent_news = get_recent_news(symbol)
            display_recent_news(recent_news)

            # Display portfolio tracker
            portfolio_tracker()

            # Display technical indicators
            display_technical_indicators(stock_data)
        except Exception as e:
            st.error(f"Error: {e}")
    if __name__ == "__main__":
     main()    


    

def show_about():
    st.subheader("About")
    display_lottie_animation("about")
    st.write("About page content goes here.")
    st.subheader("Welcome to [Stock Market Prediction].")
    st.write("Welcome to [Stock Market Prediction], your trusted source for comprehensive stock market analysis, news, and insights. Whether you're a seasoned investor or just starting your journey in the world of finance, we're here to provide you with the tools and information you need to make informed decisions and navigate the complexities of the stock market.")
    st.subheader("Our Mission:")
    st.write("At [Stock Market Prediction], our mission is to empower individuals with the knowledge and resources to achieve their financial goals. We believe in democratizing access to financial information and fostering a community of informed investors who can confidently navigate the stock market.")
    st.subheader("What We Offer:")
    st.write("Market Analysis: Stay ahead of market trends with our in-depth analysis and expert commentary on key sectors, stocks, and economic indicators.")
    st.write("News and Updates: Get real-time updates on market news, earnings reports, and major events affecting the stock market.")
    st.write("Research Tools: Access powerful research tools and data analytics to conduct thorough research and identify investment opportunities.")
    st.write("Educational Resources: Learn the fundamentals of investing, trading strategies, and financial planning through our comprehensive educational resources and tutorials.")
    st.write("Community Engagement: Join our vibrant community of investors to share ideas, ask questions, and collaborate with like-minded individuals.")
    st.subheader("Meet Our Team:")
    st.write("Our team of experienced analysts, researchers, and financial experts is dedicated to providing you with timely and insightful content to help you succeed in the stock market. Get to know the faces behind [Website Name] and learn more about their expertise and passion for finance.")
    st.subheader("Our History:")
    st.write("Founded in [Year], [Stock Market Prediction] has quickly become a trusted destination for investors seeking reliable market insights and analysis. Over the years, we've grown our platform and expanded our offerings to better serve our growing community of users.")  
    st.subheader("Our Values:")
    st.write("Integrity: We are committed to upholding the highest standards of integrity and ethics in everything we do.")
    st.write("Accuracy: We strive to deliver accurate and reliable information to our users, ensuring they can make informed decisions with confidence.")
    st.write("Transparency: We believe in transparency and openness, providing clear and honest communication with our users at all times.")

def logout():

    # Perform logout actions here
    display_lottie_animation("logout")
    st.session_state.logged_in = False
    st.success("You have been logged out.")

def main():
    st.title("Stock Market Prediction")
    # Create a connection to the SQLite database
    conn = create_connection("user_db.sqlite")

    

    # Create a users table if it doesn't exist
    with conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY,
                            username TEXT NOT NULL,
                            email TEXT NOT NULL,
                            password TEXT NOT NULL
                        )''')

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    url = "https://lottie.host/b1bbf192-fb29-4199-9a7a-60bc8b875067/uAPR21HfI6.json"
    response = requests.get(url)
    animation_json = response.json()

    # Display Lottie animation only on the login page
    if st.session_state.logged_in == False:
      st_lottie(animation_json, speed=3, width=700, height=300, key="lottie_animation")


    # Sidebar options based on login status
    if st.session_state.logged_in:
        option = st.selectbox("Choose an option:", ("Market Data","Screener", "Tools", "Technical Analysis",  "About", "Logout"))
    else:
        option = st.sidebar.selectbox("Choose an option:", ("Signup","Login"))

    # Check the option selected
    if option == "Market Data":
        show_home()
    elif option == "Screener":
        show_screener()
    elif option == "Tools":
        show_tools()
    elif option == "Technical Analysis":
        show_technical_analysis()
    elif option == "About":
        show_about()
    
    elif option == "Logout":
        logout()
    elif option == "Login":
        # Add login functionality here
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            # Check if the user exists in the database
            user = check_user(conn, email, password)
            if user:
                st.session_state.logged_in = True
                st.success(f"Welcome, {user[1]}!")
                # Redirect to the home page
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")

        pass
    elif option == "Signup":
        # Add signup functionality here
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Sign Up"):
            if password == confirm_password:
                if 4 <= len(password) <= 8:
                    # Create the user in the databas
                    user_id, error_message = create_user(conn, username, email, password)
                    if user_id:
                        st.success("User created successfully!")
                    else:
                        st.error(error_message)
                else:
                    st.error("Password length must be between 4 and 8 characters.")
            else:
                st.error("Passwords do not match!")
                

if __name__ == "__main__":
    main()
