
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



def prepare_dataset(ds, 
            data_augmentation,
            resize_and_rescale,
            augment=False
            ):
    '''
    A helper function to augment and resize,reshaping a given image
    '''
    import tensorflow as tf
    batch_size = 32
    AUTOTUNE = tf.data.AUTOTUNE

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=AUTOTUNE)
    
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)


    
    return ds.prefetch(buffer_size=AUTOTUNE)

def visulize_batch_from_dataset(ds,classes_list):
    '''
    A helper function to visulize a batch of images from a given dataset
    '''
    import tensorflow as tf
    import matplotlib.pyplot as plt
    f, axs = plt.subplots(5, 7, sharey=True)
    f.set_size_inches(30,10)
    row = 0
    col = 0
    for x,y in ds:
        for batch in range (0, 32):
            try:
                axs[row][col].imshow(x[batch]/255.)
                axs[row][col].set_title(classes_list[tf.math.argmax(y[batch])])
                axs[row][col].axis("off")
                col = col+1
                if col == 7:
                    row = row+1
                    col = 0
            except:
                axs[row][col].axis("off")
        axs[4][4].axis("off")
        axs[4][5].axis("off")
        axs[4][6].axis("off")
        break
    return None

def plot_model_hist(hist_var):
    '''
    A helper function to plot loss and accuracy from history variable
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=1, ncols=2)
    pd.DataFrame(hist_var.history)[["loss","val_loss"]].plot(title="Loss Plots",ax=axes[0],figsize=(10,5))
    pd.DataFrame(hist_var.history)[["accuracy","val_accuracy"]].plot(title="Accuracy Plots",ax=axes[1],figsize=(10,5))
    return None


def prepare_tb_call_back(dir_name, experiment_name):
    import tensorflow as tf
    from datetime import datetime
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_call_back = tf.keras.callbacks.TensorBoard(
                                    log_dir=log_dir,
                                )
    print("Saving TensorBoard log files to: " , log_dir)
    return tb_call_back

def prepare_checkpoint_call_back(dir_name, experiment_name):
    import tensorflow as tf
    filepath = dir_name + "/" + experiment_name 
    ckpt_call_back = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=filepath,
                                                monitor = 'val_loss',
                                                verbose=1,
                                                save_best_only = False,
                                                save_weights_only = False,
                                                mode = 'auto',
                                                save_freq='epoch',
                                                options=None,
                                                initial_value_threshold=None,
                                            )
    print("Saving Checkpoints to files : " , filepath)
    return ckpt_call_back