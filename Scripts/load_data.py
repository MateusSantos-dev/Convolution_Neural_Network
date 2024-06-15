import tensorflow_datasets as tfds
import tensorflow as tf


def convert_label_to_fibonacci_binary(image, label) -> tuple:
    """Converte o label para 1 se for um dígito da sequência de Fibonacci e 0 caso contrário."""
    fibonacci = tf.constant([0, 1, 2, 3, 5, 8], dtype=tf.int64)
    label = tf.cast(label, tf.int64)
    is_fibonacci = tf.reduce_any(tf.equal(label, fibonacci))
    binary_label = tf.cast(is_fibonacci, tf.int32)
    return image, binary_label


def load_data(batch_size: int = 128, shuffle_buffer_size: int = 1000, binary_classification=False) -> tuple:
    """Carrega o dataset MNIST e retorna os datasets de treino e teste."""
    mnist_data = tfds.load(name="mnist", as_supervised=True)
    mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]

    if binary_classification:
        mnist_train = mnist_train.map(convert_label_to_fibonacci_binary)
        mnist_test = mnist_test.map(convert_label_to_fibonacci_binary)

    mnist_train = mnist_train.shuffle(shuffle_buffer_size).batch(batch_size)
    mnist_test = mnist_test.batch(batch_size)

    return mnist_train, mnist_test
