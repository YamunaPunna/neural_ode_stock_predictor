import streamlit as st
import torch
import numpy as np
import os
from models.neural_ode_model import ODEFunc, NeuralODE
from utils.data_loader import fetch_stock_data
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(page_title="Neural ODE Stock Predictor", layout="centered")
st.title("📈 Neural ODE Stock Price Predictor")

# Input Features
st.subheader("🔍 Input Features")
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
ticker = st.selectbox("Choose a stock ticker", tickers)
user_date = st.date_input("Select date", value=datetime.now().date())
user_time = st.time_input("Select time", value=datetime.strptime("10:00", "%H:%M").time())
predict_button = st.button("Predict")

if predict_button:
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        st.error("❌ Could not fetch data.")
    else:
        prices = torch.tensor(df['Close'].values[-60:], dtype=torch.float32).view(-1, 1)
        t = torch.linspace(0, len(prices) - 1, len(prices))

        odefunc = ODEFunc()
        model = NeuralODE(odefunc)

        model_path = f"models/{ticker}_neural_ode.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            st.success("✅ Loaded pre-trained model.")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(100):
                optimizer.zero_grad()
                pred_y = model(prices[0], t)
                loss = torch.mean((pred_y - prices) ** 2)
                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), model_path)
            st.success("✅ Model trained and saved.")

        with torch.no_grad():
            pred = model(prices[0], t).squeeze().numpy()

        # Plotting Actual vs Predicted
        st.subheader("📊 Predicted vs Actual Stock Prices")
        fig, ax = plt.subplots()
        ax.plot(df.index[-60:], prices.numpy(), label="Actual")
        ax.plot(df.index[-60:], pred, label="Predicted", linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Predict for user-selected date and time
        st.subheader("📌 Selected Date & Time Prediction")
        last_known_time = df['Date'].iloc[-1] if 'Date' in df.columns else datetime.now()
        target_datetime = datetime.combine(user_date, user_time)
        hours_ahead = (target_datetime - last_known_time).total_seconds() / 3600
        future_index = int(len(prices) + hours_ahead)

        if hours_ahead < 0:
            st.warning("⚠️ Cannot predict past data. Please select a future date/time.")
        else:
            custom_t = torch.linspace(0, future_index, future_index + 1)
            with torch.no_grad():
                future_pred = model(prices[0], custom_t).squeeze().numpy()
                custom_price = future_pred[-1]

            formatted_time = target_datetime.strftime("%A, %d %B %Y at %I:%M %p")
            st.success(f"📌 Predicted price for {ticker} on {formatted_time}: **${custom_price:.2f}**")
