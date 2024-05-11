import streamlit as st # type: ignore
from datetime import date

import yfinance as yf # type: ignore
from prophet import Prophet # type: ignore
from prophet.plot import plot_plotly # type: ignore
from plotly import graph_objs as go # type: ignore

START = "2007-12-18"
TODAY = date.today().strftime("%Y-%m-%d")

selected_stock = 'ITMG.JK'
#st.image("https://itmg.co.id/img/logo.png", width=200)
centered_image_html = """
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://itmg.co.id/img/logo.png" width="200">
</div>
"""
st.markdown(centered_image_html, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Prediction Stock ITMG <br> By Teguh Yuhono</h1>", unsafe_allow_html=True)
msft = yf.Ticker(selected_stock)

with st.expander("About Company"):
  st.write(msft.info["longBusinessSummary"])
  st.metric("sector", msft.info["sector"])
  st.metric("industry", msft.info["industry"])
  st.metric("website", msft.info["website"])
  st.metric("marketCap", "{:,.0f} Bio IDR".format(msft.info["marketCap"]/ 1e9))
with st.expander("Annual Report"):
  st.write("income_stmt",msft.income_stmt)
  st.write("balance_sheet",msft.balance_sheet)
  st.write("cashflow",msft.cashflow)
with st.expander("Rekomendasi"):
  st.write(msft.recommendations_summary)
n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365  


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
#data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
# data_load_state.text('Loading data... done!')

# st.subheader('Raw data')
# st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
# st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
