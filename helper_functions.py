
def walk_through_data(raw_data_path:str):
    '''
    A helper function to wlak through the input raw data (typical kaggle style input raw data)
    '''
    import os
    from glob import glob
    for fold in os.listdir(raw_data_path):
        print(fold , "details : ")
        print("Number of classes : ", len(glob(raw_data_path+"/"+fold+"/*")))
        for cls in os.listdir(raw_data_path + "/" + fold):
            print("Class" , cls , "has : ", len(glob(raw_data_path+"/"+fold+"/"+cls+"/*")) , "files")
        print("-"*35)
    return None

def visulize_multi_class_raw_images(raw_data_path:str, fold: str):
    '''
    A helper function to visulize random raw images from a multi class usecase.
    It can show upto 10 different classes. If you have more than 10 classes, you need to tweek the funciton
    '''
    import tensorflow as tf
    from glob import glob
    import os
    import random
    import matplotlib.pyplot as plt

    columns = int(len(glob(raw_data_path + "/" + fold+"/*"))/2)
    f, axs = plt.subplots(2, columns, sharey=True)
    f.set_size_inches(30,10)

    row = 0
    col = 0
    for cls in os.listdir(raw_data_path + "/" + fold):
        sample_img = random.sample(glob(raw_data_path+"/"+fold+"/"+cls+"/*"), k=1)[0]
        sample_img = tf.io.read_file(sample_img)
        sample_img = tf.image.decode_image(sample_img)
        axs[row][col].imshow(sample_img)
        axs[row][col].set_title(cls + " - " + str(sample_img.shape))
        axs[row][col].axis("off")
        col = col+1
        if col == columns:
            row = 1
            col = 0
    return None

def average_img_shape(raw_data_path:str, fold: str):
    '''
    A helper function to get average of all images shpaes in a given folder
    '''
    import tensorflow as tf
    from glob import glob
    import os
    import numpy as np
    list_of_shapes = []
    for cls in os.listdir(raw_data_path + "/" + fold):
        for img in glob(raw_data_path+"/"+fold+"/"+cls+"/*"):
            sample_img = img
            sample_img = tf.io.read_file(sample_img)
            sample_img = tf.image.decode_image(sample_img)
            list_of_shapes.append(sample_img.shape.as_list())

    return (np.mean(np.array(list_of_shapes),axis=0))