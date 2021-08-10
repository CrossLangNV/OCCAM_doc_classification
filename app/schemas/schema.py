from typing import Dict, Optional

from pydantic import BaseModel


class ModelBase(BaseModel):
    id: int
    name: str
    description: str


class Model(ModelBase):
    pass


class ModelsInfo(BaseModel):
    models: Dict[str, Model]


class Prediction(BaseModel):
    name: str
    description: Optional[str] = None

    certainty: float
    prediction: bool
    label: Optional[str] = None
