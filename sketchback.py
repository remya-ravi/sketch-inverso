from __future__ import print_function
import numpy as np
import cv2 as cv
import os
import time
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

from skimage.transform import resize


from keras.preprocessing import image
from copy import deepcopy
#from scipy.misc import imresize
from PIL import Image
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import np_utils, plot_model
from keras import objectives, layers
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from keras import backend as K

from keras.optimizers import SGD
from math import log10, sqrt
from skimage.measure import compare_ssim
from test import test_faces
np.random.seed(1337)  # for reproducibility


# CelebA Faces: 72x88 200K Images
# ZuBuD Buildings: 120x160 3K Images
# CUHK Faces: 80x112 88 Images


'''m=189
n=150
'''
m = 176#189            #504 #176    #168, 174    282   252
n =168#150 '''           #400 #168  #176, 150      205   200
sketch_dim = (m,n)
sketch_dim2=(n,m)
img_dim = (m,n,3)
num_images = 150
num_epochs = 5
batch_size = 2
file_names = []

#CelebA_SKETCH_PATH = '~/Project/CelebA_Sketch'
#CelebA_IMAGE_PATH = '~/Project/img_align_celeba'
CelebA_SKETCH_PATH = '~/Sketchback-master/CelebA_Sketch'
CelebA_IMAGE_PATH = '~/Sketchback-master/img_align_celeba'

#BUILDING_SKETCH_PATH = 'CUHK_Sketch'
#BUILDING_IMAGE_PATH = 'CUHK'
BUILDING_SKETCH_PATH = 'CelebA_Sketch1'
BUILDING_IMAGE_PATH = 'img_align_celeba1'

CUHK_SKETCH_PATH = '~/Sketchback-master/CUHK_Sketch'
CUHK_IMAGE_PATH = '~/Sketchback-master/CUHK'


base_model = vgg16.VGG16(weights='imagenet', include_top=False)
vgg = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)


def load_file_names(path):
    return os.listdir(path)

'''def sub_plot(x,y,z):
    fig = plt.figure()
    a = fig.add_subplot(1,3,1)
    imgplot = plt.imshow(x, cmap='gray')
    a.set_title('Sketch')
    plt.axis("off")
    a = fig.add_subplot(1,3,2)
    imgplot = plt.imshow(z)
    a.set_title('Prediction')
    plt.axis("off")
    a = fig.add_subplot(1,3,3)
    imgplot = plt.imshow(y)
    a.set_title('Ground Truth')
    plt.axis("off")
    plt.show()'''

def imshow(x, gray=False):
    plt.imshow(x, cmap='gray' if gray else None)
    plt.show()


def get_batch(idx, X = True, Y = True, W = True, dataset='zubud'):
    
    global file_names

    X_train = np.zeros((batch_size, m, n), dtype='float32')
    Y_train = np.zeros((batch_size, m, n, 3), dtype='float32')
    F_train = None
    
    if dataset == 'zubud':
        x_path = BUILDING_SKETCH_PATH
        y_path = BUILDING_IMAGE_PATH
    elif dataset == 'cuhk':
        x_path = CUHK_SKETCH_PATH
        y_path = CUHK_IMAGE_PATH
    else:
        x_path = CelebA_SKETCH_PATH
        y_path = CelebA_IMAGE_PATH
    
    if len(file_names) == 0:
        file_names = load_file_names(x_path)
        
    if X:
        # Load Sketches
        for i in range(batch_size):
            file = os.path.join(x_path, file_names[i+batch_size*idx])
            img = cv.imread(file,0)
            img = resize(img, sketch_dim)
            #img=numpy.array(Image.fromarray(arr).resize(img, sketch_dim))
            img = img.astype('float32')
            X_train[i] = img 
            #/ 255.
            
    if Y:
        # Load Ground-truth Images
        for i in range(batch_size):
            file = os.path.join(y_path, file_names[i+batch_size*idx])
            img = cv.imread(file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = resize(img, img_dim)
            img = img.astype('float32')
            Y_train[i] = img #/ 255.
    
    if W:
        F_train = get_features(Y_train)
    
    X_train = np.reshape(X_train, (batch_size, m, n, 1))
    
    return X_train, Y_train, F_train



def get_features(Y):
    Z = deepcopy(Y)
    Z = preprocess_vgg(Z)
    features = vgg.predict(Z, batch_size = 5, verbose = 0)
    return features

def preprocess_vgg(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    x = 255. * x
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] = x[:, 0, :, :] - 103.939
        x[:, 1, :, :] = x[:, 1, :, :] - 116.779
        x[:, 2, :, :] = x[:, 2, :, :] - 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] = x[:, :, :, 0] - 103.939
        x[:, :, :, 1] = x[:, :, :, 1] - 116.779
        x[:, :, :, 2] = x[:, :, :, 2] - 123.68
    return x

def preprocess_VGG(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    #print(dim_ordering)
    # assert dim_ordering in {'tf','th'}
    # x has pixels intensities between 0 and 1
    x = 255. * x
    norm_vec = K.variable([103.939, 116.779, 123.68])
    if dim_ordering == 'th':
        norm_vec = K.reshape(norm_vec, (1,3,1,1))
        x = x - norm_vec
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        norm_vec = K.reshape(norm_vec, (1,1,1,3))
        x = x - norm_vec
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def feature_loss(y_true, y_pred):
    #print("inside FL........")
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def pixel_loss(y_true, y_pred):
    #print("inside PL........")
    return K.sqrt(K.mean(K.square(y_true - y_pred))) + 0.00001*total_variation_loss(y_pred)

'''def adv_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)'''

def total_variation_loss(y_pred):
    #print("inside TVL........")
    if K.image_data_format() == 'channels_first':
        a = K.square(y_pred[:, :, :m - 1, :n - 1] - y_pred[:, :, 1:, :n - 1])
        b = K.square(y_pred[:, :, :m - 1, :n - 1] - y_pred[:, :, :m - 1, 1:])
    else:
        a = K.square(y_pred[:, :m - 1, :n - 1, :] - y_pred[:, 1:, :n - 1, :])
        b = K.square(y_pred[:, :m - 1, :n - 1, :] - y_pred[:, :m - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def generator_model(input_img):

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.add([x, res])

    # Decoder
    res = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(encoded)
    x = layers.add([encoded, res])
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])

    x = Conv2D(128, (2, 2), activation='relu', padding='same', name='block6_conv1')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block7_conv1')(x)
    res = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])

    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='block8_conv1')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(x)
    res = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])

    x = Conv2D(32, (2, 2), activation='relu', padding='same', name='block10_conv1')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block11_conv1')(x)
    res = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return decoded

def feat_model(img_input):
    # extract vgg feature
    vgg_16 = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None)
    # freeze VGG_16 when training
    for layer in vgg_16.layers:
        layer.trainable = False
    
    vgg_first2 = Model(inputs=vgg_16.input, outputs=vgg_16.get_layer('block2_conv2').output)
    Norm_layer = Lambda(preprocess_VGG)
    x_VGG = Norm_layer(img_input)
    feat = vgg_first2(x_VGG)
    return feat

def full_model(summary = True):
    input_img = Input(shape=(m, n, 1))
    generator = generator_model(input_img)
    feat = feat_model(generator)
    #model = Model(input=input_img, output=[generator, feat], name='architect')
    model= Model(name="image",inputs=input_img,outputs=[generator, feat])
    model.summary()
    return model

def train_faces(weights=None):
    gen_start_time = time.time()
    model = full_model()
    optim = Adam(lr=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=[pixel_loss, feature_loss], loss_weights=[1, 0.01], optimizer=optim , metrics=['accuracy','mse']) #mae, mse
    #opt = SGD(lr=0.01, momentum=0.9)
    #model.compile(loss='mean_squared_error', optimizer=optim)
    
    if weights is not None:
        model.load_weights(weights)
    print(model.metrics_names)
    print("...................................................\nEpoch...Batch....:Loss....:Accuracy....:MSE....")
    for epoch in range(num_epochs):
        num_batches = num_images // batch_size
        
        for batch in range(num_batches):
            X,Y,W = get_batch(batch, dataset='zubud')
            #loss=model.fit(X,[Y,W],batch=num_batches)
            #print("......loss....",loss)
           # hist = model.fit(X,[Y,W] , validation_split=0.2)
            loss = model.train_on_batch(X, [Y, W])
            #tes=model.test_on_batch(X,[Y,W])
            print("Train : Epoch#",epoch,"|Batch#", batch, ":", loss[0], ":",loss[3], ":",loss[4])
            #print("Test:Epoch#",epoch,"|Batch#", batch, ":", tes[0], ":",tes[3], ":",tes[4])

        model.save_weights("weights",overwrite=True)
    
    #print(model.evaluate(X,[Y,W], verbose=0))
    test_faces()
    print("time taken: ", time.time() - gen_start_time)

    model.save("model_" + str(time.time()) + ".h5")

def evalfn(weights=None):
    gen_start_time = time.time()
    model = full_model()
    model.compile(loss='mae', optimizer='adam' , metrics=['accuracy','mse']) #mae, mse
    
    if weights is not None:
        model.load_weights(weights)
    
    X,Y,W = get_batch(1, dataset='zubud')
            
    history = model.fit(X, [Y,W] , epochs=5,batch_size=5)
    print("accuracy: " + str(acc()))
    
def acc(y_true, y_pred):
    return np.equal(y_true, np.round(y_pred)).mean()
            

'''def predict(batch, i, weights):
    model = full_model()
    model.load_weights(weights)
    X, T, _ = get_batch(batch, Y = True, W = False, dataset='zubud')
    
    Y, W = model.predict(X[:i])
    x = X[i].reshape(m,n)
    y = Y[i]
    
    sub_plot(x, T[i], y)'''

def load_image(img_path, show=False):

    img = image.load_img(img_path,grayscale=True, target_size=(m, n))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


def sketchback(image, weights):
    model = full_model()
    model.load_weights(weights)


    sketch = cv.imread(image, 0)
    sketch = cv.resize(sketch, sketch_dim)
    original=sketch
    
    #sketch=numpy.array(Image.fromarray(arr).resize(sketch, sketch_dim))
    #cv.imshow("img",sketch)
    #sketch = sketch / 255
    
    #sketch = sketch.reshape(1,m,n,1)
    sketch = load_image(image)
    result, _ = model.predict(sketch)
    #imshow(result[0])
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(sketch[0].reshape(m,n), cmap='gray')
    a.set_title('Sketch')
    plt.axis("off")
    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(result[0])
    a.set_title('Prediction')
    plt.axis("off")
    plt.show()

    '''original=resize(original,img_dim)
    outimg=result[0]
    outimg=resize(outimg,img_dim)
    #cv.imshow("img",outimg)
    value = PSNR(original, outimg) 
    #print(f"PSNR value is {value} dB")

    (score, diff) = compare_ssim(original, outimg, full=True,multichannel=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM:")
    print("Score=",score)'''
    
       
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    print("MSE=",mse)
    print("rmse=",sqrt(mse))
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 10 * log10(max_pixel / sqrt(mse)) 
    psnr=psnr/3
    return psnr 
'''def get_img_size():
    from PIL import Image

    a = os.listdir("CUHK")
    b = os.listdir("CUHK_Sketch")
    result = []

    for i,j in zip(a,b):
        e = Image.open("CUHK/"+i)
        d = Image.open("CUHK_Sketch/"+j)
        width1, height1 = e.size
        width2, height2 = d.size

        if not width1==width2 and height1==height2:
            result.append(i)
    #print(result)'''





    
if __name__ == "__main__":
    #sketchback("face05.jpg", "weights_faces")
    train_faces()
    #evalfn()
    # get_img_size()'''
