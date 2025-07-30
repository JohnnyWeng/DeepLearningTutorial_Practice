import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
torch.manual_seed(18)

data = yf.download("2330.TW", start="2021-01-01", end="2024-01-01")
# Close: 收盤價
prices = data['Close'].values.astype(float)

# Step 2: Preprocess the data
scaler = MinMaxScaler(feature_range=(-1, 1))
prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
seq_length = 30  # Use past 30 days to predict the next day
x_data, y_data = create_sequences(prices_normalized, seq_length)

# Split into training and testing data
train_size = int(len(x_data) * 0.8)
# 800 以後 -> test data (200 data)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initial value of hidden layer: h0: 0
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the models
# Instantiate the models and move them to device
rnn_model = RNNModel().to(device)
from torchinfo import summary
print(summary(rnn_model))

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the models
lstm_model = LSTMModel().to(device)
print(summary(lstm_model))
# Set up optimizers and loss function
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.01)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.05)
criterion = nn.MSELoss()


# Training function
def train_model(model, optimizer, x_train, y_train, epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


print("Training RNN Model...")
train_model(rnn_model, rnn_optimizer, x_train, y_train)

print("Training LSTM Model...")
train_model(lstm_model, lstm_optimizer, x_train, y_train)
# Evaluation function
def evaluate_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        loss = criterion(predictions, y_test)
    return predictions, loss.item()

# Evaluate both models
rnn_predictions, rnn_loss = evaluate_model(rnn_model, x_test, y_test)
lstm_predictions, lstm_loss = evaluate_model(lstm_model, x_test, y_test)

print(f'RNN Test Loss: {rnn_loss}')
print(f'LSTM Test Loss: {lstm_loss}')
# Inverse transform the predictions to original scale
rnn_predictions = scaler.inverse_transform(rnn_predictions.cpu().numpy())
lstm_predictions = scaler.inverse_transform(lstm_predictions.cpu().numpy())
y_test_actual = scaler.inverse_transform(y_test.cpu().numpy())

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Price")
plt.plot(rnn_predictions, label="RNN Predictions")
plt.plot(lstm_predictions, label="LSTM Predictions")
plt.legend()
plt.title("RNN vs LSTM Predictions on Google Stock Prices")
plt.show()

