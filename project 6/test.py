# import pandas as pd
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import yfinance as yf
# from statsmodels.tsa.arima.model import ARIMA
# import warnings
#
# matplotlib.use("TkAgg")
# # Завантаження даних про акції Apple з Yahoo Finance
# data = yf.download('AAPL', start='2010-01-01', end='2023-12-31')
#
# # Використовуємо тільки закриті ціни акцій для аналізу
# data = data['Close']
#
# # Переконуємося, що індекс даних має правильний формат і частоту
# data.index = pd.to_datetime(data.index)
# data = data.asfreq('B')  # 'B' означає бізнес-дні
#
# # Розділяємо дані на тренувальний та тестовий набори
# train_data = data[:int(0.9 * len(data))]
# test_data = data[int(0.9 * len(data)):]
#
# # Створення моделі ARIMA
# model = ARIMA(train_data, order=(5, 1, 0))  # order=(p, d, q)
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     model_fit = model.fit()
#
# # Вивід резюме моделі
# print(model_fit.summary())
#
# # Передбачення цін на акції
# predictions = model_fit.forecast(steps=len(test_data))
# predictions = pd.Series(predictions, index=test_data.index)
# print(predictions)
#
# # Побудова графіків
# plt.figure(figsize=(12, 6))
# plt.plot(train_data, label='Тренувальні дані')
# plt.plot(test_data.index, test_data, label='Тестові дані', color='red')
# plt.plot(test_data.index, predictions, label='Передбачені дані', color='green')
# plt.legend(loc='upper left')
# plt.title('Передбачення цін на акції Apple за допомогою ARIMA')
# plt.xlabel('Дата')
# plt.ylabel('Ціна закриття')
# plt.show()
# import yfinance as yf
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import matplotlib
#
# matplotlib.use("TkAgg")
#
#
# def download_data(ticker, period='5y'):
#     stock_data = yf.download(ticker, period=period)
#     return stock_data
#
#
# def prepare_data(stock_data: pd.DataFrame):
#     # Використовуємо лише стовпчик 'Close'
#     stock_data = stock_data[['Close']]
#     stock_data['Prediction'] = stock_data['Close'].shift(-30)  # Прогнозуємо ціну через 30 днів
#
#     # Створюємо набір даних X (всі дані, окрім останніх 30 днів)
#     X = stock_data.drop(['Prediction'], axis=1)[:-30]
#     # Створюємо набір даних y (ціни через 30 днів)
#     y = stock_data['Prediction'][:-30]
#
#     return X, y
#
#
# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Стандартизуємо дані
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     # Створюємо і тренуємо модель
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#
#     return model, scaler, X_test, y_test
#
#
# def predict_future_prices(model, scaler, stock_data):
#     # Дані для прогнозування (останні 30 днів)
#     prediction_data = stock_data[['Close']][-30:]
#     prediction_data = scaler.transform(prediction_data)
#
#     # Прогнозуємо ціни
#     future_prices = model.predict(prediction_data)
#     return future_prices
#
#
# def plot_results(stock_data, future_prices):
#     stock_data['Prediction'] = stock_data['Close'].shift(-30)
#
#     # Додаємо прогнозовані ціни до даних
#     stock_data['Future_Prediction'] = stock_data['Prediction']
#     stock_data['Future_Prediction'][-30:] = future_prices
#
#     plt.figure(figsize=(14, 7))
#     plt.plot(stock_data['Close'], label='Actual Price')
#     plt.plot(stock_data['Future_Prediction'], label='Predicted Price')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.show()
#
#
# def main():
#     ticker = 'AAPL'
#     stock_data = download_data(ticker)
#
#     X, y = prepare_data(stock_data)
#     model, scaler, X_test, y_test = train_model(X, y)
#
#     future_prices = predict_future_prices(model, scaler, stock_data)
#     plot_results(stock_data, future_prices)
#
#
# if __name__ == '__main__':
#     main()

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
    # Використовуємо лише стовпчик 'Close'
    data = stock_data['Close'].values.reshape(-1, 1)

    # Масштабуємо дані для моделі LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Визначаємо параметри для навчальних даних
    prediction_days = 60

    X_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler


def train_model(X_train, y_train):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=25, batch_size=32)

    return model


def predict_future_prices(model, scaler, stock_data, prediction_days=60):
    # Дані для прогнозування (останні 60 днів)
    data = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    X_test = []
    for x in range(prediction_days, len(scaled_data)):
        X_test.append(scaled_data[x-prediction_days:x, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions


def plot_results(stock_data, predictions, prediction_days=60):
    train = stock_data['Close'][:-prediction_days]
    valid = stock_data['Close'][-prediction_days:]
    valid = valid.reset_index()
    valid['Predictions'] = predictions[-prediction_days:]

    plt.figure(figsize=(14, 7))
    plt.plot(train, label='Training Data')
    plt.plot(valid['Close'], label='Actual Price')
    plt.plot(valid['Predictions'], label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    ticker = 'AAPL'
    stock_data = download_data(ticker)

    X_train, y_train, scaler = prepare_data(stock_data)
    model = train_model(X_train, y_train)

    predictions = predict_future_prices(model, scaler, stock_data)
    plot_results(stock_data, predictions)


if __name__ == '__main__':
    main()
