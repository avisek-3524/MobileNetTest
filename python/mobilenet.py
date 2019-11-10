#importing modules

import keras
from keras import backend as k
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D,Activation
from keras.applications import MobileNet,VGG16
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix




#taking the pretrained model from keras
# mobile = keras.applications.mobilenet.MobileNet()

# #preparing an array out of a image
# def prepareImage(file):
#     imgPath = ""
#     img = image.load_img(imgPath + file, target_size=(224,224))
#     imgArray = image.img_to_array(img)
#     imgArrayExpandedDims = np.expand_dims(imgArray,axis = 0)
#     return keras.applications.mobilenet.preprocess_input(imgArrayExpandedDims)
# """
# Image(filename="./data/a.jpg")
# preprocessed_image = prepareImage("./data/a.jpg")
# predictions = mobile.predict(preprocessed_image)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)

# """

# base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dense(1024,activation='relu')(x) #dense layer 2
# x=Dense(512,activation='relu')(x) #dense layer 3
# preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

# model=Model(inputs=base_model.input,outputs=preds)
# #specify the inputs
# #specify the outputs
# #now a model has been created based on our architecture

# #making last few layers as untrained 
# for layer in model.layers:
#     layer.trainable=False

# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

# train_generator1=train_datagen.flow_from_directory('./data/angelE',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)

# train_generator2=train_datagen.flow_from_directory('./data/davidE',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)

# train_generator3=train_datagen.flow_from_directory('./data/fridayE',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)

# train_generator4=train_datagen.flow_from_directory('./data/jarvisE',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)

# train_generator5=train_datagen.flow_from_directory('./data/not',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)

# X_type_1 = np.array(train_generator1)
# X_type_2 = np.array(train_generator2)
# X_type_3 = np.array(train_generator3)
# X_type_4 = np.array(train_generator4)
# X_type_5 = np.array(train_generator5)

# train_generator = np.concatenate((X_type_1, X_type_2, X_type_3, X_type_4,X_type_5), axis=0)


# #Remember our color pixels, the ones that went from 0-255? We'll rescale them to be between 0 and 1. This will make the model work better.
# train_generator = train_generator / 255.




# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy

# step_size_train=train_generator.n//train_generator.batch_size
# model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    epochs=10)


# base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dense(1024,activation='relu')(x) #dense layer 2
# x=Dense(512,activation='relu')(x) #dense layer 3
# preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


# model=Model(inputs=base_model.input,outputs=preds)
# #specify the inputs
# #specify the outputs
# #now a model has been created based on our architecture

# for layer in model.layers:
#     layer.trainable=False

# train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

# train_generator=train_datagen.flow_from_directory('/home/shawfamily/Desktop/mobileNetTest/python/data/input',
#                                                  target_size=(224,224),
#                                                  color_mode='rgb',
#                                                  batch_size=32,
#                                                  class_mode='categorical',
#                                                  shuffle=True)

# model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# # Adam optimizer
# # loss function will be categorical cross entropy
# # evaluation metric will be accuracy

# step_size_train=train_generator.n//train_generator.batch_size
# model.fit_generator(generator=train_generator,
#                    steps_per_epoch=step_size_train,
#                    epochs=10)


# def load_image(img_path, show=False):

#     img = image.load_img(img_path, target_size=(150, 150))
#     img_tensor = image.img_to_array(img)                    # (height, width, channels)
#     img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
#     img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

#     if show:
#         plt.imshow(img_tensor[0])                           
#         plt.axis('off')
#         plt.show()

#     return img_tensor
  
# #img_path = 'C:/Users/Ferhat/Python Code/Workshop/Tensoorflow transfer learning/blue_tit.jpg'
# img_path = '/home/shawfamily/Desktop/mobileNetTest/python/data/download.jpeg'
# new_image = load_image(img_path)

# pred = model.predict(new_image)




train_path = "data/train"
test_path = "data/test"
valid_path = "data/valid"
classs = ['Elephant1','Elephant2','Elephant3','Elephant4','Not Registered']
train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path, target_size=(224,224), classes=classs, batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path, target_size=(224,224), classes=classs, batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path, target_size=(224,224), classes=classs, batch_size=10, shuffle=False)

#build and train css
# model = Sequential([
#     Conv2D(32, (3,3),activation = 'relu',input_shape = (224,224,3)),
#     Flatten(),
#     Dense(2,activation='softmax'),
# ])

# model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
# model.fit_generator(train_batches,steps_per_epoch = 4, validation_data = valid_batches, validation_steps = 4, epochs = 5, verbose = 2)

# #predicting 
# #predictions = model.predict_generator(test_batches, steps = 1, verbose = 0)
# #transfer learning

mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()

x = mobile.layers[-6].output
predictions = Dense(len(classs),activation='softmax')(x)
model = Model(inputs = mobile.input, outputs = predictions)
# model = Sequential()
for layer in model.layers[:-5]:
    layer.trainable = False

#training the model
model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
model.fit_generator(train_batches,steps_per_epoch = 4, validation_data = valid_batches, validation_steps = 2, epochs = 20, verbose = 2)

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor
  
#img_path = 'C:/Users/Ferhat/Python Code/Workshop/Tensoorflow transfer learning/blue_tit.jpg'
img_path = '/home/shawfamily/Desktop/mobileNetTest/python/data/ad.jpeg'
new_image = load_image(img_path)

pred = model.predict(new_image)
print(pred)
item = max(pred[0])
print(item)
# imdexs = pred[0].indices(item)
# print(imdexs)
for i,j in enumerate(pred):
    for k,l in enumerate(j):
        if(l==item):
            print(i,k)
print(f"type : {classs[k]}")


