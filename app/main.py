from typing import List

from PIL import Image
from fastapi import UploadFile, File, FastAPI, Header

from app.schemas.schema import ModelsInfo, Prediction, ModelSecret
from classifier.methods import get_pred
from classifier.models import DocModel
from models.config import ModelConfig
from scripts.machine_readable import _p_machine_readable, scanned_document

THRESHOLD = .5
B_MR = False  # Disabled to avoid confusion
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "FASTAPI for the microservice: document classification."}


@app.get("/models", response_model=ModelsInfo)
async def get_models():
    d = {'models': ModelConfig.MODELS}

    return d


@app.post("/classify", response_model=Prediction)
async def post_classify(model_id: int = Header(...),
                        file: UploadFile = File(...),
                        ):
    """

    :return:
    """

    def _get_model(model_id: int) -> ModelSecret:
        for model in filter(lambda model: model.id == model_id, ModelConfig.MODELS):
            return model

        raise ValueError(f'Unexpected value for model: {model_id}')

    fast_model = _get_model(model_id)

    keras_model = DocModel()
    keras_model.load_weights(fast_model.filename)
    # keras_model =         tf.keras.models.load_model(model.filename)

    im = Image.open(file.file)
    p1 = get_pred(im, keras_model)

    # TODO find out if there is a need to release the GPU memory manually.
    # from tensorflow.keras import backend as K
    # K.clear_session()

    # prediction = p1 >= .5
    #
    # label = 'BRIS' if prediction else 'NBB'

    prediction = (p1 >= .5)
    label = fast_model.get_label() if prediction else fast_model.get_not_label()

    p = Prediction(
        name=fast_model.name,
        description=fast_model.description,
        certainty=p1,
        prediction=prediction,
        label=label,
    )

    return p


@app.post("/classify/multiple",
          response_model=List[Prediction]
          )
async def post_classify_multiple(model_id: int = Header(...),
                                 files: List[UploadFile] = File(...),
                                 ):
    """
    :return:
    """

    l = []
    for file in files:
        prediction = await post_classify(model_id=model_id,
                                         file=file)

        l.append(prediction)

    return l


if B_MR:
    @app.post("/machine_readable",
              response_model=Prediction
              )
    async def post_machine_readable(file: UploadFile = File(...),
                                    threshold=THRESHOLD):
        """

        Args:
            file:

        Returns:

        """

        p = _p_machine_readable(file.file)

        prediction = Prediction(name='machine readable',
                                description='Predict if a PDF is machine readable or not.',
                                certainty=p,
                                prediction=(p >= threshold))

        return prediction


@app.post("/scanned_document",
          response_model=Prediction
          )
async def post_scanned_document(file: UploadFile = File(...),
                                threshold: float = THRESHOLD):
    """ Detect if a PDF contains a scanned document, i.e. might contain non-machine readable text.

    Args:
        file: PDF

    Returns:
        Information about the prediction
    """

    p_scanned = scanned_document(file.file)

    prediction = Prediction(name='Scanned document',
                            description='Predict if a PDF is a scanned document.',
                            certainty=p_scanned,
                            prediction=(p_scanned >= threshold)
                            )

    return prediction
