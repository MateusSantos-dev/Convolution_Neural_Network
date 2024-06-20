from fastapi import APIRouter
from typing import Any
from API.schemas import CNNInput
from Scripts.evaluate_model import evaluate_model
from Scripts.load_data import load_data
from Scripts.train_model import train_model

network_router = APIRouter()


def validate_input(network: CNNInput) -> None:
    """Valida os parâmetros de entrada da rede neural."""
    if network.classification_type not in ["binary", "multiclass"]:
        raise ValueError("classification_type must be 'binary' or 'multiclass'")
    if network.epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    if network.optmizer not in ["Adam", "SGD"]:
        raise ValueError("optimizer must be 'Adam' or 'SGD'")
    if network.classification_type == "binary" and network.layers[-1]["units"] != 2:
        raise ValueError("For binary classification, the output layer must have 2 units")
    if network.classification_type == "multiclass" and network.layers[-1]["units"] != 10:
        raise ValueError("For multiclass classification, the output layer must have 10 units")


@network_router.post("/treinar", response_model=Any)
async def test(network: CNNInput) -> Any:
    """Treina e avalia um modelo de rede neural."""
    validate_input(network)
    training_data, test_dataset = load_data(
        binary_classification=True if network.classification_type == "binary" else False
    )
    modelo = train_model(training_dataset=training_data, network=network)
    evaluate_model(modelo, test_dataset)
    return 1
