
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

def visulize_raw_images():
    pass