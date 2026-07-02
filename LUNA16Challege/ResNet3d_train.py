import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow.compat.v1 as tf
from pathlib import Path


from ResNet3d.model_resNet3d import ResNet3dModule
import numpy as np
import pandas as pd




def train():    
    with tf.device('/gpu:0'):
        '''
        Preprocessing for dataset
        '''
        # Read  data set (Train data from CSV file)
        # csv file should have the type:
        # label,data_npy
        # label,data_npy
        # ....
        #
        path_data = Path(__file__).parent / "dataprocess/data/training_origine.csv"
        with path_data.open() as file:
            csvimagedata = pd.read_csv(file,delimiter=',')
            data = csvimagedata.iloc[:, :].values
            np.random.shuffle(data)
            # For Image
            images = data[:, 1:]
            # For Labels
            labels = data[:, 0]
            ResNet3d = ResNet3dModule(48, 48, 48, channels=1, n_class=2, costname="cross_entropy")
            ResNet3d.train(images, labels, "resnet.pd", "log/NoudleClassfy/resnet/", 0.001, 0.7, 10, 32)


train()
