import streamlit as st 
from datetime import date
import pickle
import numpy as np
np.float_ = np.float64
import yfinance as yf
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2007-12-18"
TODAY = date.today().strftime("%Y-%m-%d")


centered_image_html = """
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://unsia.ac.id/wp-content/uploads/2022/12/logo-1.png" width="200">
</div>
"""
st.markdown(centered_image_html, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Prediction Harga Saham <br> By Teguh Yuhono</h1>", unsafe_allow_html=True)
stocks = ['ITMG.JK', 'BBRI.JK', 'BMRI.JK', 'BBCA.JK']
# Single stock selection (Dropdown)
selected_stock = st.selectbox('Select a stock:', stocks)
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

	
data = load_data(selected_stock)

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

m = pickle.load(open('rnn_model_by_teguh.sav', 'rb'))
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
