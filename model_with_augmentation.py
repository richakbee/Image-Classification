
#steps1-extract data from zip file to a tmp folder
import os
from os import getcwd
import zipfile
#srep2-inside the folder extract data
#step3-create directories for training and validation and  sub directories in training & validation
# folders to take advantage of automatic class labelling
#step4-split data now from source to tarining & testing folders(do random shuffle first)
import utility_functions #split_data function in this takes care of step 3 & 4
#step5-plot some random data from each class labels from source
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#step6- define model & create imagedatagenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#step7-train model ,plot accuracy (both training & validation)
#step8-optimize the model if possible using augmentation then plot again(same as step 7)

#step0
#checking for GPUS
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#to find which device the operations are assigned to
#tf.debugging.set_log_device_placement(True)
basedir=getcwd()
print("base directory=",basedir)
#step1
path_datasets=f"{basedir}/datasets"
path_cats_dogs_zip=f"{path_datasets}/cats-and-dogs.zip"
#
#extract the data in the datasets directory
#does it only when ran for first time
import pathlib
file = pathlib.Path(os.path.join(path_datasets,'PetImages'))
if file.exists ():
    pass
else:
    local_zip=path_cats_dogs_zip
    zip_ref=zipfile.ZipFile(local_zip,'r')
    zip_ref.extractall(path_datasets)
    zip_ref.close()

print("checking for the extracted folder named PetImages is created",os.listdir(path_datasets))


source_data=os.path.join(path_datasets,'PetImages')
print('path to source data',source_data)

split_size=0.9


train_image_count,val_image_count=utility_functions.image_count(source_data,split_size)
print('{} for training and {} for testing'.format(
                            train_image_count,
                            val_image_count))


#create & split data
file = pathlib.Path(os.path.join(path_datasets,'Training'))
if file.exists ():
    train_path=os.path.join(path_datasets,'Training')
    val_path=os.path.join(path_datasets,'Validation')
else:
    train_path,val_path=utility_functions.split_data(source_data,'Training','Validation',split_size)

#step5
utility_functions.show_image_from_each_class('D:\PythonProjects\ImageClassification/datasets\PetImages',4)

#step6
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

#step7
#set up data generators that will read pictures in our source folders,
#convert them to float32 tensors, and feed them (with their labels) to our network.

train_datagen=ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                shear_range=0.2,
                height_shift_range=0.2,
                width_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                zoom_range=0.2
                )

val_datagen=ImageDataGenerator(
                rescale=1./255
                )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_path,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  val_datagen.flow_from_directory(val_path,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=train_image_count/20,
                              epochs=15,
                              validation_steps=val_image_count/20,
                              verbose=2)

# model.save("model_with_augmentation.h5")
# print("Saved model to disk")

model.save_weights('model_aug.weights.h5')
print("saved model weights")

#plot accuracy & loss of the model
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']


epochs=range(len(acc))


fig,axs=plt.subplots(2)
axs[0].plot(epochs , acc , 'r',label='Accuracy')
axs[0].plot(epochs , val_acc ,'b', label='Validation Accuracy')
axs[0].legend()
fig.suptitle("Accuracy plot")

axs[1].plot(epochs, loss, 'r', label='Loss')
axs[1].plot(epochs, val_loss ,label='Validation Loss')
axs[1].legend()


plt.show()

# import pathlib
# cwd=getcwd()

# model_aug_path=pathlib.Path(os.path.join(cwd,'model_with_augmentation.h5'))
# if model_aug_path.exists():
#     print(os.stat(model_aug_path).st_size/2**20 ,"MB")