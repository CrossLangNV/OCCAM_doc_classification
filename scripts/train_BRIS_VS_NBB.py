import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from classifier.datasets import Training, Subset, BOG, NBB
from classifier.methods import FILENAME_MODEL
from classifier.models import DocModel
from scripts.train_DH import gen_y_from_x


def get_data(b_scratch: bool = False, subset: Subset = Subset.TRAIN) -> (np.ndarray, np.ndarray):
    x_BOG = BOG(subset)
    x_NBB = NBB(subset)

    y_BOG = gen_y_from_x(x_BOG, 1)
    y_NBB = gen_y_from_x(x_NBB, 0)

    x = np.concatenate([x_BOG, x_NBB], axis=0)
    y = np.concatenate([y_BOG, y_NBB], axis=0)

    return x, y


def main(filename_model=None,
         ):
    net = DocModel()

    data = Training()
    # X_train, X_test, y_train, y_test = train_test_split(*data, test_size=.2, random_state=123)

    X_train, y_train = get_data(subset=Subset.TRAIN)
    X_valid, y_valid = get_data(subset=Subset.VALID)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    hist = net.fit_fast(X_train, y_train,
                        epochs=1000,
                        validation_data=(X_valid, y_valid),
                        callbacks=es)

    print(hist.history)

    if filename_model:
        net.save(filename_model)

    y_pred = net.predict(data[0])

    acc = np.mean((y_pred[:, 0] >= 0) ==
                  data[1].astype(bool)
                  )
    print(f'Accuracy on training and validation data: {acc}')

if __name__ == '__main__':
    main(FILENAME_MODEL)