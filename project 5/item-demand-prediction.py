import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, \
    QFileDialog, QDateEdit, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from keras.models import Sequential
from keras.layers import Dense, LSTM


def prepare_data(data, store_id, item_id):
    filtered_data = data[(data['store'] == store_id) & (data['item'] == item_id)]
    filtered_data = filtered_data[['date', 'sales']]
    filtered_data.set_index('date', inplace=True)

    return filtered_data


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def train_model(data, store_id, item_id):
    df = prepare_data(data, store_id, item_id)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(df.values)

    look_back = 10
    trainX, trainY = create_dataset(dataset, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

    # Створення моделі LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)

    return model, scaler, look_back


def predict_sales(model, scaler, look_back, df, future_date):
    dataset = scaler.transform(df.values)

    predictions = []
    last_data = dataset[-look_back:]
    future_dates = pd.date_range(start=df.index[-1], end=future_date)

    for date_ in future_dates:
        input_data = np.reshape(last_data, (1, look_back, 1))
        prediction = model.predict(input_data)
        predictions.append(prediction[0][0])
        last_data = np.append(last_data[1:], prediction)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return pd.DataFrame(data={'date': future_dates, 'sales': predictions.flatten()})


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class SalesPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Прогнозування попиту на товар')

        self.layout = QVBoxLayout()

        self.upload_button = QPushButton('Виберіть файл з даними про попит')
        self.upload_button.clicked.connect(self.upload_data)
        self.layout.addWidget(self.upload_button)

        self.date_label = QLabel('Дата:')
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDate(date.today())

        self.store_id_label = QLabel('ID магазину:')
        self.store_id_input = QComboBox(self)

        self.item_id_label = QLabel('ID товару:')
        self.item_id_input = QComboBox(self)

        self.predict_button = QPushButton('Передбачити попит', self)
        self.predict_button.clicked.connect(self.predict_sales)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        self.layout.addWidget(self.date_label)
        self.layout.addWidget(self.date_input)
        self.layout.addWidget(self.store_id_label)
        self.layout.addWidget(self.store_id_input)
        self.layout.addWidget(self.item_id_label)
        self.layout.addWidget(self.item_id_input)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)

    def upload_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Виберіть файл з даними про попит', '', 'CSV Файли (*.csv)')
        if file_path:
            self.data = pd.read_csv('train.csv')

            self.data['date'] = pd.to_datetime(self.data['date'])
            for store_id in self.data["store"].unique():
                self.store_id_input.insertItem(store_id - 1, str(store_id))
            for item_id in self.data["store"].unique():
                self.item_id_input.insertItem(item_id - 1, str(item_id))

    @staticmethod
    def set_button_text(button, text):
        button.setText(text)

    def predict_sales(self):
        self.predict_button.setText("Прогнозування...")
        date = self.date_input.text()
        store_id = int(self.store_id_input.currentText())
        item_id = int(self.item_id_input.currentText())

        try:
            df = prepare_data(self.data, store_id, item_id)
            model, scaler, look_back = train_model(self.data, store_id, item_id)
            predicted_df = predict_sales(model, scaler, look_back, df, date)
            prediction = predicted_df[predicted_df['date'] == date]['sales'].values[0]

            self.canvas.axes.clear()
            self.canvas.axes.plot(df.index, df['sales'], label='Попит на товар')
            self.canvas.axes.plot(predicted_df['date'], predicted_df['sales'],
                                  label='Прогнозований попит', linestyle='--')
            self.canvas.axes.axvline(x=pd.to_datetime(date), color='r', linestyle=':', label='Дата прогнозу')
            self.canvas.axes.set_xlabel('Дата')
            self.canvas.axes.set_ylabel('Попит')
            self.canvas.axes.set_title(f'Попит на товар {item_id} в магазині {store_id}')
            self.canvas.axes.legend()
            self.canvas.axes.grid(True)
            self.predict_button.setText("Передбачити попит")
            self.canvas.draw()
            QMessageBox.information(self, 'Прогноз', f'Прогнозований попит: {prediction: .2f}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SalesPredictorApp()
    ex.show()
    sys.exit(app.exec_())
