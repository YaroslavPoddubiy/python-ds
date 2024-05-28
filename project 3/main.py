import numpy as np
import cv2
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import sys
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi


matplotlib.use("TkAgg")

ssd_model = tf.saved_model.load("model")


def detect(image_path, label_path):
    labels = load_labels(label_path)

    image_np = np.array(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

    detections = detect_objects(image_np)

    visualize(image_np, detections, labels)


def detect_objects(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = ssd_model(input_tensor)

    return detections


def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split()[0]): ' '.join(line.split()[1:]) for line in f.readlines()}
    return labels


def visualize(image_np, detections, labels):
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    image_with_detections = image_np.copy()

    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            box = detection_boxes[i]
            y1, x1, y2, x2 = box
            (left, right, top, bottom) = (x1 * image_np.shape[1], x2 * image_np.shape[1],
                                          y1 * image_np.shape[0], y2 * image_np.shape[0])

            cv2.rectangle(image_with_detections, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            class_name = labels[detection_classes[i]]
            label = f'{class_name}: {detection_scores[i]:.2f}'
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            cv2.rectangle(image_with_detections, (int(left), int(top - label_size[1])),
                          (int(left + label_size[0]), int(top + base_line)), (0, 255, 0), cv2.FILLED)
            cv2.putText(image_with_detections, label, (int(left), int(top)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    plt.figure()
    plt.imshow(image_with_detections)
    plt.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("uis/MainWindow.ui", self)

        self.setWindowTitle("Виберіть фото")

        self.pushButton.clicked.connect(self.open_file_window)

    def open_file_window(self):
        fname = QFileDialog(self).getOpenFileName(self, 'Open file',
                                                  "D:/images", "Image files (*.jpg *.jpeg *.jfif)")
        detect(str(fname[0]), "model/labels.txt")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
