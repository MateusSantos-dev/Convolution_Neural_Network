import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from API.schemas import CNNInput
from keras import layers, models, callbacks, regularizers, optimizers
from typing import List, Dict, Any, Optional


class LossHistory(callbacks.Callback):
    """Callback para salvar o histórico de perda e acurácia do modelo."""

    def on_train_begin(self, logs=None) -> None:
        """Cria o arquivo de histórico de perda e acurácia."""
        with open("Outputs/loss_history.txt", "w") as arquivo:
            arquivo.write("Inicio do treinamento\n")

    def on_epoch_end(self, epoch, logs=None) -> None:
        """Salva a perda e acurácia do modelo a cada época."""
        with open("Outputs/loss_history.txt", "a") as arquivo:
            arquivo.write(f"Epoca {epoch + 1} - Acuracia: {logs['accuracy']}, Erro: {logs['loss']}\n")


def save_weights(model: models.Sequential, filename: str, path: str) -> None:
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


def save_model(model: models.Sequential, filename: str, path: str) -> None:
    """
    Salva o modelo no diretório especificado.
    Arquivo com nome {filename}.keras
    """
    os.makedirs(path, exist_ok=True)
    final_path = os.path.join(path, f"{filename}.keras")
    model.save(final_path)


def create_model(network_layers: List[Dict[str, Any]]):
    """Cria um modelo de rede neural convolucional com base nas camadas especificadas."""
    model = models.Sequential()
    try:
        model.add(layers.InputLayer(shape=(28, 28, 1)))
        for layer in network_layers:
            if layer["type"] == "conv":
                model.add(layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel_size"],
                    activation=layer["activation"]))
            elif layer["type"] == "pool":
                model.add(layers.MaxPooling2D(pool_size=layer["pool_size"]))
            elif layer["type"] == "flatten":
                model.add(layers.Flatten())
            elif layer["type"] == "dense":
                lambda_l2 = layer.get("lambda_l2", 0)
                model.add(layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                    kernel_regularizer=regularizers.l2(lambda_l2) if lambda_l2 > 0 else None))
            elif layer["type"] == "dropout":
                model.add(layers.Dropout(rate=layer["rate"]))
            else:
                raise ValueError(f"Tipo de camada desconhecido: {layer['type']}"
                                 f"Escolha entre 'conv', 'pool', 'flatten', 'dense' e 'dropout'")
    except KeyError as e:
        raise ValueError(f"Campo obrigatório não encontrado: {e}")
    return model


def create_optimizer(optimizer_name: str, learning_rate: Optional[float]):
    """Cria um otimizador com base no nome e taxa de aprendizado especificados"""
    if optimizer_name == "Adam":
        return optimizers.Adam(learning_rate=learning_rate if learning_rate else 0.001)
    elif optimizer_name == "SGD":
        return optimizers.SGD(learning_rate=learning_rate if learning_rate else 0.01)
    else:
        raise ValueError(f"Otimizador não reconhecido: {optimizer_name}"
                         f"Escolha entre 'Adam' e 'SGD'")


def create_early_stopping(early_stopping: Optional[Dict[str, Any]]) -> Optional[callbacks.EarlyStopping]:
    """Cria um callback de parada antecipada com base nas configurações especificadas."""
    if early_stopping is None:
        return None
    return callbacks.EarlyStopping(
        min_delta=early_stopping["min_delta"],
        patience=early_stopping["patience"],
        restore_best_weights=True,
        verbose=2
    )


def get_validation_dataset(
        training_dataset: tf.data.Dataset,
        validation_split: float,
        batch_size: int
) -> tf.data.Dataset:
    """Retorna um dataset de validação com base no dataset de treino e no split especificado."""
    training_size = 60000  # tamanho do dataset MNIST
    validation_size = int(training_size * validation_split)
    validation_dataset = training_dataset.take(validation_size // batch_size)
    return validation_dataset


def plot_two_metrics(
        history: callbacks.History,
        metric1: str,
        metric2: str,
        title: str,
        xlabel: str,
        ylabel: str,
        legend: str | List[str]
) -> None:
    """Plota dois métricas de um histórico de treinamento."""
    plt.plot(history.history[metric1])
    plt.plot(history.history[metric2])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc="upper left")
    plt.show()


def train_model(
        training_dataset: tf.data.Dataset,
        network: CNNInput,
        batch_size: int = 128
) -> models.Sequential:
    """Treina a rede neural especificada e salva os resultados."""
    validation_dataset = get_validation_dataset(
        training_dataset,
        validation_split=0.1,
        batch_size=batch_size
    )

    # Definindo a arquitetura da rede neural
    model = create_model(network.layers)
    optimizer = create_optimizer(network.optmizer, network.learning_rate)
    model.compile(optimizer=optimizer, loss=network.loss_function, metrics=["accuracy", network.loss_function])
    model.summary()

    # Salvando modelo
    current_path = os.path.dirname(__file__)
    models_path = os.path.join(current_path, '..', 'Models')
    save_hyperparameters(network.model_dump(mode="json"), models_path)
    save_weights(model, "pesos_iniciais", models_path)

    parada_antecipada = create_early_stopping(network.early_stopping)
    if parada_antecipada is not None:
        callbacks = [parada_antecipada, LossHistory()]
    else:
        callbacks = [LossHistory()]

    # Treinando o modelo
    history = model.fit(
        training_dataset, epochs=network.epochs,
        batch_size=batch_size,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=2
    )

    # Salvando pesos finais e modelo
    save_weights(model, "pesos_finais", models_path)
    save_model(model, "modelo", models_path)

    # Acurácias de treino e validação
    plot_two_metrics(
        history,
        "accuracy",
        "val_accuracy",
        "Acurácia do Modelo",
        "Epoca",
        "Acurácia",
        ["Treino", "Validação"])
    plt.show()

    # Erros de treino e validação
    plot_two_metrics(
        history,
        "loss",
        "val_loss",
        "Erro do Modelo",
        "Epoca",
        "Erro",
        ["Treino", "Validação"])
    plt.show()

    return model
