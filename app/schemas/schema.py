from typing import Optional, List

from pydantic import BaseModel


class ModelBase(BaseModel):
    id: int
    name: str
    description: str
    label: Optional[str] = None  # If the prediction is True
    not_label: Optional[str] = None  # If the prediction is False, else

    def get_label(self):
        if self.label:
            return self.label
        else:
            return 'label 1'

    def get_not_label(self):
        if self.not_label:
            return self.not_label

        else:
            return f'not {self.get_label()}'


class Model(ModelBase):
    pass


class ModelSecret(ModelBase):
    """
    Should not be shared with users.
    """
    filename: str  # filename to weights of model.


class ModelsInfo(BaseModel):
    models: List[Model]


class Prediction(BaseModel):
    name: str
    description: Optional[str] = None

    certainty: float
    prediction: bool
    label: Optional[str] = None
