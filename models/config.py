import os

from app.schemas.schema import ModelSecret

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class ModelConfig:
    """
    Configure and add your own models
    """
    MODELS = [
        ModelSecret(id=1,
                    name='BOG vs. NBB',
                    description='Classifier that distinguishes '
                                'Belgian Official Gazette (BOG) from National Bank of Belgium (NBB) documents.\n'
                                'True if BOG, False if NBB',
                    filename=os.path.join(ROOT, 'models/model_nbb_bris.h5'),
                    label='BOG',
                    not_label='NBB'
                    ),
        ModelSecret(id=2,
                    name='Digital Humanities',
                    description='Detect documents from the Digital Humanities.',
                    filename=os.path.join(ROOT, 'models/model_DH.h5'),
                    label='DH',
                    ),
    ]

# from enum import unique, Enum
# @unique
# class ModelConfig(Enum):
#     """
#     Configure and add your own models
#     """
#     BRIS = ModelSecret(id=1,
#                        name='BOG vs. NBB',
#                        description='Distinguishes Belgian BRIS documents from NBB.',
#                        filename=os.path.join(ROOT, 'models/model_nbb_bris.h5'),
#                        label='BOG',
#                        not_label='NBB'
#                        )
#     DH = ModelSecret(id=2,
#                      name='Digital Humanities',
#                      description='Detect documents from the Digital Humanities.',
#                      filename=os.path.join(ROOT, 'models/model_DH.h5'),
#                      label='DH',
#                      )
