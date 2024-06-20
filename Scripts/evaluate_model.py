import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras import models


def evaluate_model(model: models.Sequential, test_dataset: tf.data.Dataset) -> None:
    """Avalia o modelo com o dataset de teste e salva as previsões."""
    metrics = model.evaluate(test_dataset, verbose=2)
    print(f"Test Loss: {metrics[0]}")
    print(f"Test Accuracy: {metrics[1]}")

    # Previsões no dataset de teste
    test_predictions = model.predict(test_dataset)
    test_labels = []
    for _, label in test_dataset:
        test_labels.extend(label.numpy())
    test_predictions = tf.argmax(test_predictions, axis=1)

    # Salvando previsões
    current_path = os.path.dirname(__file__)
    outputs_path = os.path.join(current_path, '..', 'Outputs')
    prediction_path = os.path.join(outputs_path, "predictions.csv")
    np.savetxt(prediction_path, test_predictions.numpy(), delimiter=",", fmt="%d")

    # Matriz de confusão
    matriz_confusao = confusion_matrix(test_labels, test_predictions)
    ConfusionMatrixDisplay(matriz_confusao).plot()
    matrix_path = os.path.join(outputs_path, "matriz_confusao.png")
    plt.savefig(matrix_path)
    plt.show()
