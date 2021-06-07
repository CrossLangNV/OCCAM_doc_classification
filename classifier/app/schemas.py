from typing import Dict, List

from pydantic import BaseModel


class ModelBase(BaseModel):
    id: int
    name: str
    description: str


class Model(ModelBase):
    pass


class ModelsInfo(BaseModel):
    models: Dict[str, Model]


class PredictionBase(BaseModel):
    idx: int
    certainty: List[float]
    label: str


class Prediction(PredictionBase):
    pass
