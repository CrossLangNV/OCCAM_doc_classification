import numpy as np

from classifier.datasets import BRIS, DH


def gen_y_from_x(x: np.ndarray, label:int):
    return label*np.ones((x.shape[0], ), dtype=np.int)


def get_data() -> (np.ndarray, np.ndarray):

    x_BRIS = BRIS() # np.concatenate([x_BOG, x_NBB])
    y_BRIS = gen_y_from_x(x_BRIS, 0)

    x_DH = DH()
    y_DH = gen_y_from_x(x_DH, 1)

    x = np.concatenate([x_BRIS, x_DH], axis=0)
    y = np.concatenate([y_BRIS, y_DH], axis=0)

    return x, y

def main():
    x, y = get_data()


    return

if __name__ == '__main__':
    main()