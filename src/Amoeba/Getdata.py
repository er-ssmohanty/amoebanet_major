from Amoeba.ImportAll import *

def get_data(train_dir,validation_dir,batch_size=64,test_train_split=0.2):
    train_data = image_dataset_from_directory(\
      train_dir,color_mode="grayscale",image_size=(256,256) ,\
      subset='training',seed=12, validation_split=test_train_split,\
      batch_size=batch_size)
    validation_data = image_dataset_from_directory(validation_dir,
      color_mode="grayscale",image_size=(256,256), subset='validation',seed=12,\
      validation_split=test_train_split,batch_size=batch_size)
    return train_data,validation_data
