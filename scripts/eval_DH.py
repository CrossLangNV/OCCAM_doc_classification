import numpy as np
from sklearn.metrics import confusion_matrix

from classifier.datasets import Subset
from classifier.models import DocModel

from scripts.train_DH import FILENAME_MODEL_DH, get_data


def main(verbose=1):
    net = DocModel()
    net.load_weights(FILENAME_MODEL_DH)

    X_test, y_test = get_data(subset=Subset.TEST)

    y_pred = net.predict(X_test)
    y_pred_label = np.greater_equal(y_pred, 0)

    conf = confusion_matrix(y_test, y_pred_label)

    if verbose:
        print('Confusion matrix Digital Humanities (DH) vs non-DH:')
        print(conf)

    return conf

    return

if __name__ == '__main__':
    main()