from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class CNNInput(BaseModel):
    classification_type: str
    layers: List[Dict[str, Any]]
    epochs: int
    loss_function: str
    optmizer: str
    learning_rate: Optional[float] = Field(default=None, examples=[0.001, 0.01])
    early_stopping: Optional[Dict[str, Any]] = Field(default=None, examples=[{"min_delta": 0.001, "patience": 5}])
