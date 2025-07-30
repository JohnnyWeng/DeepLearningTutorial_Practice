import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
df = pd.read_csv('AirPassengers.csv')
print(df.head(10))
print(df.tail(10))


# Example #2
# Data Visualization
df2 = df.set_index('Month')
df2.plot(legend=None)
plt.xticks(rotation=30)
plt.show()


dataset = df2[['#Passengers']].values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print('dataset1.shape = ', dataset.shape) # (144,1)
print('before scalar:', dataset[0:5], dataset[-5:])

look_back = 2 # 用兩筆資料預測下一筆資料
def create_dataset(data, look_back):
    x, y = [], []
    for i in range(len(data) - look_back):
        _x = data[i:(i + look_back)] # 0:2
        _y = data[i + look_back] # 0+2
        x.append(_x)
        y.append(_y)
    return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))


train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train_data, test_data = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)
print('trainX.shape, trainY.shape, testX.shape, testY.shape = ', trainX.shape, trainY.shape, testX.shape, testY.shape)

# Reshape the data for "LSTM input"
trainX = trainX.reshape(-1, look_back, 1).to(device)
testX = testX.reshape(-1, look_back, 1).to(device)
trainY = trainY.to(device)
testY = testY.to(device)

# EXAMPLE of LSTM 5
# Define LSTM Model
class TimeSeriesModel(nn.Module):
    def __init__(self, look_back, hidden_size=4, num_layers=1):
        super(TimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # The critical model here.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x.size(0) -> 94
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        return self.fc(out[:, -1, :])
model = TimeSeriesModel(look_back, hidden_size=5, num_layers=1).to(device)


print(model)

# EXAMPLE of LSTM 6
model = TimeSeriesModel(look_back, hidden_size=5, num_layers=1).to(device)
print(model)

# Train the model
num_epochs = 2000
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(trainX)
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}")
train_model()

# EXAMPLE of LSTM 7
# Make predictions
model.eval()
trainPredict = model(trainX).cpu().detach().numpy()
testPredict = model(testX).cpu().detach().numpy()
print('trainY.shape, trainPredict.shape = ', trainY.shape, trainPredict.shape)
# Same as above (48-2)
print('testY.shape, testPredict.shape = ', testY.shape, testPredict.shape)


trainPredict = scaler.inverse_transform(trainPredict)
trainY_actual = scaler.inverse_transform(trainY.cpu().detach().numpy().reshape(-1, 1))
testPredict = scaler.inverse_transform(testPredict)
testY_actual = scaler.inverse_transform(testY.cpu().detach().numpy().reshape(-1, 1))

# Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainY_actual, trainPredict))
print(f'Train RMSE: {trainScore:.2f}')
testScore = math.sqrt(mean_squared_error(testY_actual, testPredict))
print(f'Test RMSE: {testScore:.2f}')

# MinMaxScaler()
from sklearn import preprocessing
import numpy as np
X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
scaler= preprocessing.MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
print(X)
print(X_scaled)
X1=scaler.inverse_transform(X_scaled)
print(X1)

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
test_start = len(trainPredict) + (look_back * 2)
testPredictPlot[test_start:test_start + len(testPredict), :] = testPredict

# Plot the results
plt.figure(figsize=(10, 6))
# Check if reshaping is needed
if dataset.shape[1] != scaler.data_min_.shape[0]:
    dataset = dataset.reshape(-1, scaler.data_min_.shape[0])
plt.plot(scaler.inverse_transform(dataset), label='Actual')
plt.plot(trainPredictPlot, label='Train Predict')
plt.plot(testPredictPlot, label='Test Predict')
plt.legend()
plt.show()



