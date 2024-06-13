import tensorflow as tf
import numpy as np
from keras import layers, models, callbacks, regularizers, optimizers
import matplotlib.pyplot as plt
from load_data import load_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json


class LossHistory(callbacks.Callback):

    def on_train_begin(self, logs=None):
        with open('Outputs/loss_history.txt', 'w') as arquivo:
            arquivo.write('Inicio do treinamento\n')

    def on_epoch_end(self, epoch, logs=None):
        with open('Outputs/loss_history.txt', 'a') as arquivo:
            arquivo.write(f'Epoca {epoch + 1} - Erro: {logs["loss"]}, Acuracia: {logs["accuracy"]}\n')


if __name__ == '__main__':
    batch_size = 128
    initial_learning_rate = 0.01
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
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(l2_regularization)),
        layers.Dropout(rate=dropout_rate),
        layers.Dense(units=10, activation='softmax')
    ])

    parada_antecipada = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=2,
        restore_best_weights=True,
        verbose=2
    )
    optimizer = optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Salvando modelo
    model.save_weights("Models/pesos_iniciais.weights.h5")
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
    with open('Outputs/hyperparameters.json', 'w') as file:
        json.dump(hyperparameters, file)

    history = model.fit(
        training_dataset, epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_dataset,
        callbacks=[parada_antecipada, LossHistory()],
        verbose=2
    )

    model.save_weights("Models/pesos_finais.weights.h5")
    model.save("Models/modelo_final.keras")

    metrics = model.evaluate(test_dataset, verbose=2)
    print(f'Test Loss: {metrics[0]}')
    print(f'Test Accuracy: {metrics[1]}')

    # Acurácias de treino e validação
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Epoca')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()

    # Matriz de Confusão
    test_predictions = model.predict(test_dataset)
    test_labels = []
    for _, label in test_dataset:
        test_labels.extend(label.numpy())
    test_predictions = tf.argmax(test_predictions, axis=1)

    np.savetxt('Outputs/predictions.csv', test_predictions.numpy(), delimiter=',', fmt='%d')

    matriz_confusao = confusion_matrix(test_labels, test_predictions)
    ConfusionMatrixDisplay(matriz_confusao).plot()

    plt.savefig('Outputs/matriz_confusao.png')

    plt.show()
