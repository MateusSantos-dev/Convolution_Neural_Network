import tensorflow_datasets as tfds


def load_mnist(batch_size: int = 128, shuffle_buffer_size: int = 1000):
    """Carrega o dataset MNIST e retorna os datasets de treino e teste."""

    # Carregando o dataset MNIST
    mnist_data = tfds.load(name='mnist', as_supervised=True)
    mnist_train, mnist_test = mnist_data['train'], mnist_data['test']

    mnist_train = mnist_train.shuffle(shuffle_buffer_size).batch(batch_size)
    mnist_test = mnist_test.batch(batch_size)

    return mnist_train, mnist_test
