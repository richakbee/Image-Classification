import os
import shutil
import random
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def split_data(SOURCE,TRAINING_FOLDER_NAME,TESTING_FOLDER_NAME,SPLIT_SIZE):
    """ a SOURCE directory containing the files
     a TRAINING directory will be created with name TRAINING FOLDER NAME
      that a portion of the files will be copied to
     a TESTING directory will be created with name TESTING FOLDER NAME
     that a portion of the files will be copie to
     a SPLIT SIZE to determine the portion
     The files should also be randomized, so that the training set is a random
     X% of the files, and the test set is the remaining files
     :type SOURCE: object
     """
    source_path= SOURCE
    parent_path=os.path.dirname(source_path)
    train_path=os.path.join(parent_path,TRAINING_FOLDER_NAME)
    val_path = os.path.join(parent_path, TESTING_FOLDER_NAME)
    list_sub_directories=os.listdir(source_path)[1:]
    #creating new directories
    try:
        os.mkdir(train_path)
        print("{} created".format(TRAINING_FOLDER_NAME))
        os.mkdir(val_path)
        print("{} created".format(TESTING_FOLDER_NAME))

        # in training & validation create as many sub
        # folders as there is in source data :for here just 2: cat & dog
        for i in list_sub_directories:
            os.mkdir(os.path.join(train_path, i))
        print("Subdirectories created in {} folder".format(TRAINING_FOLDER_NAME))

        for i in list_sub_directories:
            os.mkdir(os.path.join(val_path, i))
        print("Subdirectories created in {} folder".format(TESTING_FOLDER_NAME))
    except OSError:
        pass
    #copying file to the directories
    print("copying files to the respective sub directories in {} and {} directories with split size of {}".format(TRAINING_FOLDER_NAME,TESTING_FOLDER_NAME,SPLIT_SIZE))
    for i in list_sub_directories:
        train_list = []
        val_list = []
        sub_folder=os.path.join(SOURCE,i)
        if(os.path.getsize(sub_folder)>0):
            #shuffle files in the subfolder
            sub_folder_files=os.listdir(sub_folder)
            random.sample(sub_folder_files,len(sub_folder_files))
            for j in sub_folder_files:
                sub_folder_file=os.path.join(sub_folder,j)
                if(os.path.getsize(sub_folder_file)>0):
                    if(len(train_list)<len(sub_folder_files)*SPLIT_SIZE):
                        train_list.append(j)
                    else:
                        val_list.append(j)
            train_sub_path=os.path.join(train_path,i)
            val_sub_path=os.path.join(val_path,i)
            for i in train_list:
                shutil.copy(sub_folder_file,os.path.join(train_sub_path,i))
            for i in val_list:
                shutil.copy(sub_folder_file,os.path.join(val_sub_path,i))

            #printing description
            #dir_names=['source',TRAINING_FOLDER_NAME,TESTING_FOLDER_NAME]
            #dir_paths=[source_path,train_path,val_path]



    print('the {} directory is created at location {} '.format(TRAINING_FOLDER_NAME,train_path))
    print('the {} directory is created at location {}'.format(TESTING_FOLDER_NAME, val_path))

    return   train_path,val_path


def show_image_from_each_class(SOURCE,NUM_IMAGES=False):
    """
    we will create ydnamic plottin function, where we can set the number of images we want to
    see from each class
    """
    source_path=SOURCE
    classes=os.listdir(source_path)[1:]
    print("{} classes found : {}".format(len(classes),' , '.join(classes)))
    total_img = len(classes) * NUM_IMAGES
    num_of_img=NUM_IMAGES
    if (NUM_IMAGES==False)or (NUM_IMAGES>3):
        NUM_IMAGES=3
    ncols=NUM_IMAGES
    nrows=int(np.ceil(total_img/ncols))
    img_list=[]
    for i in classes:
        path_sub_dir=os.path.join(source_path,i)
        files_sub_dir=os.listdir(path_sub_dir)
        files_sub_dir_path=[os.path.join(path_sub_dir,fname)
                            for fname in sample(files_sub_dir,num_of_img) #selecting images at random
                            ]
        img_list.extend(files_sub_dir_path)

    fig=plt.gcf()
    #fig.subplots_adjust(hspace = .5, wspace=.01)  # setting 4 inches
    fig.set_size_inches(ncols * 4, nrows * 4)

    for i ,img_path in enumerate(img_list):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.axis('Off')
        img=mpimg.imread(img_path)
        ax.imshow(img)
    plt.show()

def image_count(source,split_size):
    size=0
    for i in os.listdir(source)[1:]:
        path=os.path.join(source,i)
        size=size+len(os.listdir(path))
    total_images=size
    print('total images = ',total_images)
    train_image_length=total_images*split_size
    test_image_length=total_images-train_image_length
    return int(train_image_length),int(test_image_length)