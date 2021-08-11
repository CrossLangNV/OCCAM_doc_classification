from typing import List

from PIL import Image
from fastapi import UploadFile, File, FastAPI, Header

from classifier.methods import get_pred_nbb_bris
from scripts.machine_readable import _p_machine_readable, scanned_document
from app.schemas.schema import Model, ModelsInfo, Prediction

THRESHOLD = .5
B_MR = False # Disabled to avoid confusion
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "FASTAPI for the microservice: document classification."}


@app.get("/models", response_model=ModelsInfo)
async def get_models():
    d = {'models': {}}

    def add_model(name: str,
                  id: int,
                  description: str):
        d.get('models')[name] = Model(id=id,
                                      description=description,
                                      name=name)

    # TODO put into a constructor
    add_model('NBB_BRIS', 1, 'Distinguishes Belgian BRIS documents from NBB')

    return d


@app.post("/classify", response_model=Prediction)
async def post_classify(model_id: int = Header(...),
                        file: UploadFile = File(...),
                        ):
    """

    :return:
    """

    if model_id == 1:
        im = Image.open(file.file)
        p1 = get_pred_nbb_bris(im)

        prediction = p1 >= .5

        label = 'BRIS' if prediction else 'NBB'
    else:
        raise ValueError(f'Unexpected value for model: {model_id}')

    p = Prediction(
        name='BOG vs. NBB',
        description='Classifier that distinguishes '
                    'Belgian Official Gazette documents from National Bank of Belgium.\n'
                    'True if BOG, False if NBB',
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
                                threshold:float=THRESHOLD):
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
