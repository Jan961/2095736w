import numpy as np
from ..config import DATA_ROOT
import os
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase




labels = """0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot"""

rng = np.random.default_rng()

# (60 000, 784)


class FmnistFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, size: int=1000):
        if os.path.exists(os.path.join(DATA_ROOT, 'fashion-mnist_train.csv')):
            XY = np.genfromtxt(os.path.join(DATA_ROOT, 'fashion-mnist_train.csv'), delimiter=",", skip_header=1)
            sample_indx = rng.choice(60000, size=size, replace=False)
            sample = XY[sample_indx]
            labels = sample[:,0]
            data = sample[:,1:]/ 255.0

        else:
            os.system('pip install tensorflow')
            from tensorflow.keras.datasets import fashion_mnist

            (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
            sample_indx = rng.choice(60000, size=size, replace=False)
            data = X_train[sample_indx]
            labels = Y_train[sample_indx]


        return data, labels

