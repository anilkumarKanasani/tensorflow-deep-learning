
def walk_through_data(raw_data_path:str):
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