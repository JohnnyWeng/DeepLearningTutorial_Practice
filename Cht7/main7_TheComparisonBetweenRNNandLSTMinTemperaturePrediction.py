import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
torch.manual_seed(18)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = pd.read_csv('temperature_data.csv')  # Ensure the file contains date and temperature columns
temperature_data = data['temperature'].values  # Extract temperature data

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_data = scaler.fit_transform(temperature_data.reshape(-1, 1)).flatten()

# Build sequence data (for splitting the data)
seq_length = 10  # Use the temperature of the previous 10 days to predict the next day
X = []
y = []
for i in range(len(temperature_data) - seq_length):
    X.append(temperature_data[i:i+seq_length])
    y.append(temperature_data[i+seq_length])

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define LSTM model
class TemperatureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(TemperatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last output of LSTM
        return out

# Define RNN model
class TemperatureRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(TemperatureRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the last output of RNN
        return out

# Initialize models, loss function, and optimizer
lstm_model = TemperatureLSTM().to(device)
rnn_model = TemperatureRNN().to(device)
criterion = nn.MSELoss()

# Training function
def train_model(model, X_train, y_train, optimizer, epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Initialize optimizers
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.05)
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.05)

# Train LSTM model
print("Training LSTM model...")
train_model(lstm_model, X_train, y_train, lstm_optimizer)

# Train RNN model
print("Training RNN model...")
train_model(rnn_model, X_train, y_train, rnn_optimizer)


# Test model and calculate average absolute error over an interval
def Test_model(model, X_test, y_test, scaler, interval=30):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i in range(interval):
            test_seq = X_test[i].unsqueeze(0)
            prediction = model(test_seq)
            predicted_temp = scaler.inverse_transform(prediction.cpu().numpy().reshape(-1, 1)).flatten()[0]
            actual_temp = scaler.inverse_transform(y_test[i].cpu().numpy().reshape(-1, 1)).flatten()[0]
            predictions.append(predicted_temp)
            actuals.append(actual_temp)

    # Calculate mean absolute error (MAE) over the interval and display predictions vs. actual values
    mae = mean_absolute_error(actuals, predictions)
    print(f'Interval Mean Absolute Error (MAE): {mae:.4f}')


# Test LSTM model performance over an interval
print("Testing LSTM model over interval...")
Test_model(lstm_model, X_test, y_test, scaler)

print("Testing RNN model over interval...")
Test_model(rnn_model, X_test, y_test, scaler)

