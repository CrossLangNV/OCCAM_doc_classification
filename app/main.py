from PIL import Image
from fastapi import UploadFile, File, FastAPI, Header

from .schemas.schema import Model, ModelsInfo, Prediction
from classifier.methods import get_pred_nbb_bris
from typing import List

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

        idx = int(p1 >= .5)

        certainty = [1 - p1, p1]
        label = 'BRIS' if idx else 'NBB'
    else:
        raise ValueError(f'Unexpected value for model: {model_id}')

    p = Prediction(
        idx=idx,
        certainty=certainty,
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
