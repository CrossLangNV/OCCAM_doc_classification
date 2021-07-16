from classifier.data import Training
from classifier.methods import FILENAME_MODEL
from classifier.models import DocModel
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

def main(b_save=True):

    net = DocModel()

    data = Training()
    X_train, X_test, y_train, y_test = train_test_split(*data, test_size=.2, random_state=123)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    hist = net.fit_fast(X_train, y_train,
                        epochs=1000,
                        validation_data=(X_test, y_test),
                        callbacks=es)

    print(hist.history)

    if b_save:
        net.model.save(FILENAME_MODEL)

    y_pred = net.predict(data[0])

    acc = np.mean((y_pred[:, 0] >= 0) ==
                  data[1].astype(bool)
                  )
    print(f'Accuracy on training and validation data: {acc}')

if __name__ == '__main__':
    main()