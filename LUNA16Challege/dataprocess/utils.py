import os


def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs

        
def files_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(files):
            print("sub_files:", files)
            return files
        
def save_file2csvv2(file_dir, file_name,label):
    """
    save file path to csv,this is for classification
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :param label:classification label
    :return:
    """
    out = open(file_name, 'w')
    sub_files = files_name_path(file_dir)
    out.writelines("class,filename" + "\n")
    for index in range(len(sub_files)):
        out.writelines(label+","+file_dir + "/" + sub_files[index] + "\n")
       

def save_file2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    sub_dirs = file_name_path(file_dir)
    out.writelines("filename" + "\n")
    for index in range(len(sub_dirs)):
        out.writelines(file_dir + "/" + sub_dirs[index] + "\n")


save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset0", "Segmentation3dImagesubset0.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset1", "Segmentation3dImagesubset1.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset2", "Segmentation3dImagesubset2.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset3", "Segmentation3dImagesubset3.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset4", "Segmentation3dImagesubset4.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset5", "Segmentation3dImagesubset5.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset6", "Segmentation3dImagesubset6.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset7", "Segmentation3dImagesubset7.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset8", "Segmentation3dImagesubset8.csv")
save_file2csv("D:/M2ISII2021/doctorat/LIDC-IDRI/LUNA 16/segmentation/Image/subset9", "Segmentation3dImagesubset9.csv")
