import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split,KFold
import sys
sys.path.append('../model_zoo/')
from evaluate import *
from augment import *
from losses import *
from images import *
from inception_resnet_unet_hypercolumns_modify import InceptionResNetV2_UNet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import SGD,Adam,RMSprop,Optimizer
from keras import Model
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


path = '../input/'

seed = 1024
np.random.seed(seed)

upsample_type = 'unet'

df_train = pd.read_csv(path+'train.csv')
ids_train = df_train['id'].map(lambda s: s.split('.')[0])
input_size = 128
epochs = 50
batch_size = 48


skf =KFold(n_splits=5, shuffle=True, random_state=1).split(ids_train)
for fold,(ind_tr, ind_te) in enumerate(skf):
    print('Training for fold:{}'.format(fold))
    ids_train_split = ids_train[ind_tr]
    ids_valid_split = ids_train[ind_te]

    #ids_train_split = ids_train_split[:30]
    #ids_valid_split = ids_valid_split[:20]
    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))


    train_data_gen_args = dict(
            rotation_range=0.,
            width_shift_range=0.0,
            height_shift_range=0.0,
            zoom_range=0,
            fill_mode='constant',
            cval=0.,
            horizontal_flip=True,
            ) 

    mask_data_gen_args = dict(
            rotation_range=0.,
            width_shift_range=0.0,
            height_shift_range=0.0,
            zoom_range=0,
            fill_mode='constant',
            cval=0.,
            horizontal_flip=True,
            ) 

    image_datagen = ImageDataGenerator(**train_data_gen_args) 
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)
    image_datagen.fit(np.zeros((1,input_size,input_size,3)), augment=False, seed=seed) 
    mask_datagen.fit(np.zeros((1,input_size,input_size,1)), augment=False, seed=seed)
    def batch_generator(imgs_train,imgs_mask_train,batch_size,seed,image_datagen,mask_datagen):
        image_generator = image_datagen.flow(imgs_train,batch_size=batch_size,shuffle=True,seed=seed)
        mask_generator = mask_datagen.flow(imgs_mask_train,batch_size=batch_size,shuffle=True,seed=seed)
    
        imgs_train_aug = []
        imgs_mask_train_aug = []
        return image_generator.next(),mask_generator.next()

    
    def train_generator(ids_train_split=ids_train_split,shuffle=True):
    
        while True:

            if shuffle:
                idx = np.arange(len(ids_train_split))
                np.random.shuffle(idx)
                ids_train_split = ids_train_split.iloc[idx]


            for start in range(0, len(ids_train_split), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(ids_train_split))
                ids_train_batch = ids_train_split[start:end]
                for id in ids_train_batch.values:

                    image_name = path+'train/images/{}.png'.format(id)
                    img = load_image(image_name,mask=False)
                    image_mask_name = path+'train/masks/{}.png'.format(id)
                    mask = load_image(image_mask_name,mask=True)
                    x_batch.append(img)
                    y_batch.append(mask)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.float32) 
                x_batch,y_batch = batch_generator(x_batch,y_batch,batch_size,seed,image_datagen,mask_datagen)

                yield x_batch, y_batch


    def valid_generator():
        while True:
            for start in range(0, len(ids_valid_split), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(ids_valid_split))
                ids_valid_batch = ids_valid_split[start:end]
                for id in ids_valid_batch.values:
                
                    image_name = path+'train/images/{}.png'.format(id)
                    img = load_image(image_name,mask=False)
                    image_mask_name = path+'train/masks/{}.png'.format(id)
                    mask = load_image(image_mask_name,mask=True)

                    x_batch.append(img)
                    y_batch.append(mask)
                x_batch = np.array(x_batch, np.float32) 
                y_batch = np.array(y_batch, np.float32) 
                yield x_batch, y_batch


    def get_val_data():

        x,y =[],[]
        for id in ids_valid_split.values:
            image_name = path+'train/images/{}.png'.format(id)
            img = load_image(image_name,mask=False)
            image_mask_name = path+'train/masks/{}.png'.format(id)
            mask = load_image(image_mask_name,mask=True)
            x.append(img)
            y.append(mask)
        return np.array(x,np.float32),np.array(y,np.float32)


    X_val,y_val = get_val_data()

    callbacks = [EarlyStopping(
                           patience=40,
                           verbose=1,
                           min_delta=1e-4,
                           ),
             ReduceLROnPlateau(factor=0.5,
                               patience=6,
                               verbose=1,
                               epsilon=1e-4,
                               ),
            feval(feval_func=['val_f1',eval_F1],save_best_weight=True,save_model_name = '../weights/best_weight_inception_resnet_unet_hypercolumns_lovasz_hinge_loss_9_19_{}_.hdf5'.format(fold),
                      X_val = X_val,
                      y_val = y_val,
                      eval_th = 0.32,
                       monitor='val_loss',
                       eval_best_only = False
                        )]
                            


    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


    inputs,outputs = InceptionResNetV2_UNet(use_activation=False)
    model = Model(inputs,outputs)



    model.compile(optimizer=RMSprop(lr=0.0002), loss=keras_lovasz_hinge)

    model.fit_generator(generator=train_generator(shuffle=True),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=1,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data= (X_val,y_val))
   
    K.clear_session() 
