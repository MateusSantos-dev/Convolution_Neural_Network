from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class CNNInput(BaseModel):
    classification_type: str
    layers: List[Dict[str, Any]]
    epochs: int
    loss_function: str
    optmizer: str
    learning_rate: Optional[float]
    early_stopping: Optional[Dict[str, Any]]
