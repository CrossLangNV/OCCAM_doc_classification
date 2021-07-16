import os
import warnings

import numpy as np
import tensorflow as tf
from PIL import Image

from classifier.methods import sigmoid, FILENAME_MODEL
from classifier.models import IMAGE_WIDTH

ROOT = os.path.join(os.path.dirname(__file__), '..')
TEST_FOLDER = os.path.join(ROOT, r'data/test/NBB')

for filename in [TEST_FOLDER]:
    if not os.path.exists(filename):
        warnings.warn(f"Couldn't find: {filename}", UserWarning)


def main():
    """

    Returns:

    """

    model_test = tf.keras.models.load_model(FILENAME_MODEL)

    results = {'label': [],
               'filename': []}

    for file in os.listdir(TEST_FOLDER):
        if file.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG')):
            im = Image.open(os.path.join(TEST_FOLDER, file))
            # Default Bicubic
            im_reshape = im.resize((IMAGE_WIDTH, IMAGE_WIDTH))
            y_pred = model_test.predict(np.array(im_reshape)[np.newaxis])

            print(f'{file} {float(y_pred):.3f} {"BRIS" if y_pred >= 0 else "NBB"}')

            results.get('label').append(float(y_pred) >= 0)
            results.get('filename').append(file)

            results.setdefault('sigmoid', []).append(sigmoid(float(y_pred)))

            print(sigmoid(y_pred))

    print(f'accuracy on test set: {np.mean(results.get("label")):.2%}')

    print(f'avg(sigmoid * label) = {np.mean(results.get("sigmoid") * 1):.2%}')

    return


if __name__ == '__main__':
    main()
