import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.neural_ode_model import ODEFunc, NeuralODE
from utils.data_loader import fetch_stock_data

# Fetch data
df = fetch_stock_data("AAPL")
prices = torch.tensor(df['Close'].values, dtype=torch.float32).view(-1, 1)
t = torch.linspace(0, len(prices)-1, len(prices))

# Prepare model
odefunc = ODEFunc()
model = NeuralODE(odefunc)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(500):
    optimizer.zero_grad()
    pred_y = model(prices[0], t)
    loss = torch.mean((pred_y - prices) ** 2)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot prediction
pred = model(prices[0], t).detach().numpy()
plt.plot(prices.numpy(), label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.title('Neural ODE Stock Prediction')
plt.show()
