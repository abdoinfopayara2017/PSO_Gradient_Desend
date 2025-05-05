from Augmentation.ImageAugmentation import DataAug3D
from glob import glob
import os


aug = DataAug3D(rotation=45, width_shift=0.05, height_shift=0.05, depth_shift=0, zoom_range=0)
for subsetindex in range(10):
    path = "E:\LUNA 16\classification\subset" + str(subsetindex) + "/1/"
    file_list = glob(path + "*.npy")
    if not os.path.exists(path+"1_aug/"):
           os.makedirs(path+"1_aug/")
    aug.DataAugmentation(file_list, 40, aug_path=path+"1_aug/")

