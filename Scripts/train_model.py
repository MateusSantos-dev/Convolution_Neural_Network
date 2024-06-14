import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras import layers, models, callbacks, regularizers, optimizers
from load_data import load_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class LossHistory(callbacks.Callback):

    def on_train_begin(self, logs=None) -> None:
        with open("Outputs/loss_history.txt", "w") as arquivo:
            arquivo.write("Inicio do treinamento\n")

    def on_epoch_end(self, epoch, logs=None) -> None:
        with open("Outputs/loss_history.txt", "a") as arquivo:
            arquivo.write(f"Epoca {epoch + 1} - Acuracia: {logs['accuracy']}, Erro: {logs['loss']}\n")


def save_weights(model: models.Model, filename: str, path: str) -> None:
    """
    Salva os pesos do modelo no diretório especificado.
    Arquivo salvo com nome {filename}.weights.h5
    """
    os.makedirs(path, exist_ok=True)
    final_path = os.path.join(path, f"{filename}.weights.h5")
    model.save_weights(final_path)


def save_hyperparameters(hyperparameters: dict, path: str) -> None:
    """
    Salva o JSON dos hiperparâmetros do modelo no diretório especificado.
    Arquivo com nome hyperparameters.json
    """
    os.makedirs(path, exist_ok=True)
    final_path = os.path.join(path, "hyperparameters.json")
    with open(final_path, "w") as file:
        json.dump(hyperparameters, file)


def save_model(model: models.Model, filename: str, path: str) -> None:
    """
    Salva o modelo no diretório especificado.
    Arquivo com nome {filename}.keras
    """
    os.makedirs(path, exist_ok=True)
    final_path = os.path.join(path, f"{filename}.keras")
    model.save(final_path)


if __name__ == "__main__":
    batch_size = 128
    initial_learning_rate = 0.0001
    dropout_rate = 0.2
    l2_regularization = 0.001
    epochs = 10
    training_dataset, test_dataset = load_mnist(batch_size)

    # dividindo o dataset de treino em treino e validação, considerando batch_size
    validation_split = 0.1
    training_size = 60000  # tamanho do dataset MNIST
    validation_size = int(training_size * validation_split)
    validation_dataset = training_dataset.take(validation_size // batch_size)

    """
    A primeira camada de convolução tem 32 filtros, a segunda tem 64 filtros e a terceira tem 64 filtros.
    Estamos usando kernel de tamanho 3x3 e função de ativação ReLU. 
    As camadas de pooling tem tamanho 2x2.
    A camada densa tem 64 neurônios e a camada de saída tem 10 neurônios porque são 10 dígitos.
    """

    # Definindo a arquitetura da rede neural
    model = models.Sequential([
        layers.InputLayer(shape=(28, 28, 1)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(units=64, activation="relu", kernel_regularizer=regularizers.l2(l2_regularization)),
        layers.Dropout(rate=dropout_rate),
        layers.Dense(units=10, activation="softmax")
    ])

    parada_antecipada = callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=2,
        restore_best_weights=True,
        verbose=2
    )
    optimizer = optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Salvando modelo
    hyperparameters = {
        "conv_layers": [
            {"filters": layer.filters, "kernel_size": layer.kernel_size, "activation": layer.activation.__name__}
            for layer in model.layers if isinstance(layer, layers.Conv2D)
        ],
        "dense_layers": [
            {"units": layer.units, "activation": layer.activation.__name__}
            for layer in model.layers if isinstance(layer, layers.Dense)
        ],
        "pooling_size": (2, 2),
        "initial_learning_rate": initial_learning_rate,
        "dropout_rate": dropout_rate,
        "l2_regularization": l2_regularization,
        "epocas": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "funcao_erro": "sparse_categorical_crossentropy",
        "parada_antecipada": {
            "min_delta": parada_antecipada.min_delta,
            "patience": parada_antecipada.patience,
        }

    }
    current_path = os.path.dirname(__file__)
    path_Models = os.path.join(current_path, '..', 'Models')
    path_Outputs = os.path.join(current_path, '..', 'Outputs')
    save_hyperparameters(hyperparameters, path_Outputs)
    save_weights(model, "pesos_iniciais", path_Models)

    # Treinando o modelo
    history = model.fit(
        training_dataset, epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_dataset,
        callbacks=[parada_antecipada, LossHistory()],
        verbose=2
    )

    # Salvando pesos finais e modelo
    save_weights(model, "pesos_finais", path_Models)
    save_model(model, "modelo", path_Models)

    # Avaliando o modelo
    metrics = model.evaluate(test_dataset, verbose=2)
    print(f"Test Loss: {metrics[0]}")
    print(f"Test Accuracy: {metrics[1]}")

    # Acurácias de treino e validação
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Acurácia do Modelo")
    plt.ylabel("Acurácia")
    plt.xlabel("Epoca")
    plt.legend(["Treino", "Validação"], loc="upper left")
    plt.show()

    # Previsões no dataset de teste
    test_predictions = model.predict(test_dataset)
    test_labels = []
    for _, label in test_dataset:
        test_labels.extend(label.numpy())
    test_predictions = tf.argmax(test_predictions, axis=1)

    prediction_path = os.path.join(path_Outputs, "predictions.csv")
    np.savetxt(prediction_path, test_predictions.numpy(), delimiter=",", fmt="%d")  # Salvando previsões

    # Matriz de confusão
    matriz_confusao = confusion_matrix(test_labels, test_predictions)
    ConfusionMatrixDisplay(matriz_confusao).plot()

    matrix_path = os.path.join(path_Outputs, "matriz_confusao.png")
    plt.savefig(matrix_path)

    plt.show()
