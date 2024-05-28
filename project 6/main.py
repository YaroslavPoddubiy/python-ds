import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from keras.models import Sequential
from keras.layers import Dense, LSTM

matplotlib.use("TkAgg")


def download_data(ticker, period='5y'):
    stock_data = yf.download(ticker, period=period)
    return stock_data


def prepare_data(stock_data: pd.DataFrame):
    data = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler


def train_model(x_train, y_train):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model


def predict(model, scaler, stock_data, prediction_days=60):
    data = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    x_test = []
    for x in range(prediction_days, len(scaled_data)):
        x_test.append(scaled_data[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions


def plot_results(stock_data, predictions, prediction_days=60):
    train = stock_data['Close'][:-prediction_days]

    stock_data['Predictions'] = 0
    stock_data["Predictions"][-prediction_days:] = predictions.reshape(-1)[-prediction_days:]

    plt.figure(figsize=(14, 7))
    plt.plot(train)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(stock_data['Close'][-prediction_days:], label='Actual Price', color="red")
    plt.plot(stock_data['Predictions'][-prediction_days:], label='Predicted Price', color="green")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    ticker = 'AAPL'
    stock_data = download_data(ticker)

    x_train, y_train, scaler = prepare_data(stock_data)
    model = train_model(x_train, y_train)

    predictions = predict(model, scaler, stock_data)
    plot_results(stock_data, predictions)


if __name__ == '__main__':
    main()
