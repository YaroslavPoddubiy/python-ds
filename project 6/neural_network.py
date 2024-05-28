import numpy as np
from random import randint
import math


# Клас для розпізнавання фігур
class NeuralNetwork:
    def __init__(self, inputs: int = 3, hidden: list | tuple | int = 36, outputs: int = 1, activation: str = "sigmoid"):
        self.__init_layers(inputs, hidden, outputs)
        self.__init_weights(inputs, hidden, outputs)
        self.__activate, self.__deactivate = self.__set_activation(activation)
        self.__total_error = None
        self.__normal_error = 0.00001

    def __init_layers(self, inputs, hidden, outputs):
        if isinstance(hidden, int):
            hidden = [hidden, ]
        self.__input_layers = 1
        self.__hidden_layers = len(hidden)
        self.__output_layers = 1
        self.__layers = [[0 for i in range(inputs)]]
        for i in range(len(hidden)):
            self.__layers.append([0 for _ in range(hidden[i])])
        self.__layers.append([0 for i in range(outputs)])
        print(self.__layers)

    # Задаємо початкові значення вагових коефіцієнтів
    def __init_weights(self, inputs, hidden, outputs):
        self.__weights = []
        for layer in range(len(self.__layers) - 1):
            self.__weights.append([[randint(0, 20) / 10 - 1 for _ in range(len(self.__layers[layer + 1]))]
                                   for i in range(len(self.__layers[layer]))])

    def __clear_layers(self):
        inputs = len(self.__inputs)
        hidden = [len(layer) for layer in self.__hidden]
        outputs = len(self.__outputs)
        self.__init_layers(inputs, hidden, outputs)

    # Функція для отримання значення виходу нейрона
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sigmoid = NeuralNetwork.sigmoid(x)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def relu(x):
        return max(0., x)

    @staticmethod
    def relu_derivative(x):
        return 1.

    @staticmethod
    def th(x):
        return math.tanh(x)

    @staticmethod
    def th_derivative(x):
        return 1 - math.tanh(x) ** 2

    def __set_activation(self, activate):
        if activate == "sigmoid":
            return self.sigmoid, self.sigmoid_derivative
        if activate == "relu":
            return self.relu, self.relu_derivative
        if activate == "th":
            return self.th, self.th_derivative
        raise ValueError("No activation function was found")

    # Функція для знаходження зваженої суми
    def sum(self, layer, neuron):
        result = 0
        prev_layer = self.__layers[layer - 1]
        if layer - 1 == 0:
            for i in range(len(prev_layer)):
                result += prev_layer[i] * self.__weights[layer - 1][i][neuron]
        else:
            for i in range(len(prev_layer)):
                result += self.__activate(prev_layer[i]) * self.__weights[layer - 1][i][neuron]

        return result

    # функція для отримання вихідних значень
    def get_output(self, inputs):
        result = []
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                self.__layers[0][j] = inputs[i][j]

            for layer in range(1, len(self.__layers)):
                for neuron in range(len(self.__layers[layer])):
                    self.__layers[layer][neuron] = self.sum(layer, neuron)

            result.append([self.__activate(value) for value in self.__layers[-1]])

        return result

    # Алгоритм зворотного поширення помилки для задання нових значень вагових коефіцієнтів
    def backward_propagation(self, inputs, result, expected, speed):
        delta_weights = self.__weights  # значення, на які потрібно змінити ваги
        output_deltas = {i: [] for i in self.__outputs}  # похибки нейронів вихідного шару
        # похибки нейронів прихованого шару
        hidden_deltas = [{i: [] for i in self.__hidden[layer]} for layer in range(len(self.__hidden))]
        # для кожного набору вхідних даних
        for i in range(len(inputs)):
            # обраховуємо сумарну функцію помилки
            total_error = sum(pow(expected[i][j] - result[i][j], 2) for j in range(len(result[i])))
            # якщо значення цієї функції зменшується до заданого значення - виходимо з циклу
            if self.__total_error and abs(total_error - self.__total_error) < self.__normal_error:
                self.__total_error = total_error
                return False
            self.__total_error = total_error
            # знаходимо похибки нейронів вихідного шару
            for j, output in enumerate(self.__outputs):
                output_deltas[output].append(
                    (expected[i][j] - self.__activate(result[i][j])) * self.__deactivate(result[i][j]))

            # знаходимо похибки нейронів прихованих шарів
            # знаходимо похибки нейронів останнього прихованого шару
            for hidden in self.__hidden[-1]:
                hidden_delta = 0
                for output in self.__outputs:
                    hidden_delta += output_deltas[output][i] * self.__weights[str(hidden) + str(output)] * \
                                    self.__deactivate(self.__hidden[-1][hidden][i])
                hidden_deltas[-1][hidden].append(hidden_delta)

            # знаходимо похибки нейронів інших прихованих шарів
            for layer in range(len(self.__hidden) - 2, -1, -1):
                for hidden in self.__hidden[layer]:
                    hidden_delta = 0
                    for hidden2 in self.__hidden[layer + 1]:
                        hidden_delta += hidden_deltas[layer + 1][hidden2][i] * self.__weights[str(hidden) + str(hidden2)] * \
                                        self.__deactivate(self.__hidden[layer][hidden][i])
                    hidden_deltas[layer][hidden].append(hidden_delta)

            # знаходимо похибки вагових коефіцієнтів, що з'єднують входи та прихований шар
            for j in self.__inputs:
                for h in self.__hidden[0]:
                    delta_weights[str(j) + str(h)].append(
                        speed * hidden_deltas[0][h][i] * self.__inputs[j][i])

            # знаходимо похибки вагових коефіцієнтів, розташованих між прихованими шарами
            for layer in range(len(self.__hidden) - 1):
                for h1 in self.__hidden[layer]:
                    for h2 in self.__hidden[layer + 1]:
                        delta_weights[str(h1) + str(h2)].append(
                            speed * hidden_deltas[layer + 1][h2][i] * self.__activate(self.__hidden[layer][h1][i])
                        )

            # знаходимо похибки вагових коефіцієнтів, що з'єднують прихований шар та вихідний
            for h in self.__hidden[-1]:
                for o in self.__outputs:
                    delta_weights[str(h) + str(o)].append(
                        speed * output_deltas[o][i] * self.__activate(self.__hidden[-1][h][i]))

        # змінюємо вагові коефіцієнти
        for weight in self.__weights:
            self.__weights[weight] += float(sum(delta_weights[weight])) / float(len(inputs))
        return True

    def test(self, inputs, outputs):
        results = self.get_output(inputs)
        results = [[self.__activate(x) for x in results[i]] for i in range(len(results))]
        return results

    # функція для тренування нейронної мережі
    def train(self, inputs, outputs, speed=0.8, iterations=2000):
        # тренування припиняється або при досягненні заданої кількості ітерацій, або якщо досягається бажане
        # значення сумарної функції помилки
        for i in range(iterations):
            result = self.get_output(inputs)
            print(result)
            results = [[self.__activate(x) for x in result[i]] for i in range(len(result))]
            if (i + 1) % 1000 == 0:
                mistake = 0
                for res, expected in zip(results, outputs):
                    mistake += sum(abs(res[i] - expected[i]) for i in range(len(res)))
                print("ITERATIONS: ", i + 1, "MISTAKE: ", round(mistake, 3))
            if not self.backward_propagation(inputs, result, outputs, speed):
                print("ITERATIONS: ", i + 1, "MISTAKE: ", self.__total_error)
                break
            self.__clear_layers()

    def __str__(self):
        return f"{self.__layers}\n{self.__weights}"


nn = NeuralNetwork(3, 5, 1)
input_dataset = [0.20, 5.14, 0.47, 4.37, 1.22, 4.29, 1.89, 4.51, 0.32, 5.80, 1.37, 5.77, 0.88, 4.86, 1.94]
input_train = [[input_dataset[i - 2] / 10, input_dataset[i - 1] / 10, input_dataset[i] / 10] for i in range(2, len(input_dataset) - 2)]
output_train = [[input_dataset[i] / 10] for i in range(3, len(input_dataset) - 1)]
nn.train(input_train, output_train, speed=0.08, iterations=1000)
print(nn)
