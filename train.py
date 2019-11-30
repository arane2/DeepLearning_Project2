import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential

#from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96

smooth = 1.

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet():
    inputs = Input((img_rows, img_cols, 1))
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(128, (3, 3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(512, (3, 3),activation='relu',padding='same'))
    model.add(Conv2DTranspose(256, (2, 2), strides=(2, 2),padding='same'))
    conv4 = Sequential()
    conv4.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    merged = concatenate([model, conv4], mode='concat')
    model2 = Sequential()
    model2.add(merged)
    model2.add(Conv2D(256,(3, 3),activation='relu',padding='same'))
    model2.add(Conv2D(256,(3, 3),activation='relu',padding='same'))
    model2.add(Conv2DTranspose(128,(2, 2),strides=(2, 2), padding='same'))
 
    conv3 = Sequential()
    conv3.add(Conv2D(128,(3, 3), activation='relu',padding='same'))
  
    merged1 = concatenate([model2, conv3], mode='concat')
    model3 = Sequential()
    model3.add(merged1)
    model3.add(Conv2D(128,(3, 3),activation='relu',padding='same'))
    model3.add(Conv2D(128,(3, 3),activation='relu',padding='same'))
    model3.add(Conv2DTranspose(64, (2, 2),strides=(2, 2), padding='same'))
    conv2 = Sequential()
    conv2.add(Conv2D(64, (3, 3),activation='relu', padding='same'))
    merged2 = concatenate([model3, conv2], mode='concat')
    model4 = Sequential()
    model4.add(merged2)
    model4.add(Conv2D(64,(3, 3),activation='relu', padding='same'))
    model4.add(Conv2D(64,(3, 3),activation='relu', padding='same'))
    model4.add(Conv2DTranspose(32, (2, 2),strides=(2, 2), padding='same'))
    conv1 = Sequential()
    conv1.add(Conv2D(32, (3, 3),activation='relu', padding='same'))
    merged3 = concatenate([model4, conv1], mode='concat')
    model5 = Sequential()
    model5.add(merged3)
    model5.add(Conv2D(32,(3, 3),activation='relu', padding='same'))
    model5.add(Conv2D(32,(3, 3),activation='relu', padding='same'))
 
    model5.add(Conv2D(1, (1, 1), activation='sigmoid'))
 
    model5.compile(optimizer=Adam(lr=2e-5), loss=dice_coef_loss, metrics=[dice_coef])
 
    return model5


def train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

def train_and_predict():
    imgs_train, imgs_mask_train = train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    model = unet()
    model.summary()
    model_checkpoint = ModelCheckpoint('/weights.h5', monitor='val_loss', save_best_only=True)

    hist = model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=100, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])
    
    xc = range(20)
    val_loss = hist.history['val_loss']
    plt.xlabel("Epochs")
    plt.ylabel("Dice coefficient Loss")
    plt.plot(xc,val_loss)
    plt.savefig('/loss.png')
    val_dice_coef = hist.history['val_dice_coef']
    plt.xlabel("Epochs")
    plt.ylabel("Dice coefficient")
    plt.plot(xc,val_dice_coef)
    plt.savefig('/dice_coef.png')

    imgs_test, imgs_id_test = test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    model.load_weights('/weights.h5')

    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('/imgs_mask_test.npy', imgs_mask_test)

    pred_dir = '/preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

if __name__ == '__main__':
    train_and_predict()
