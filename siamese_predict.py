import cv2
import numpy as np
from keras.models import load_model
import h5py
from keras import backend as K
import csv
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.models import Model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)
CUDA_VISIBLE_DEVICES=0
K.clear_session()

def load_model_(model_path):
    # weight_name = 'class5_112-0.0660_modify'
    model = load_model(model_path, compile=False)
    # model.summary()
    # global test_model
    single_model = Model(model.input, model.get_layer('dense_2').output)
    return single_model

def print_predict_value(image_name1, image_name2,single_model):
    test_model = single_model
    # image_name1 = 'kyung1_re_crop_resize'
    # image_name2 = 'up_big_crop_resize'
    #
    # img_num1 = image_name1
    # img_num2 = image_name2
    l_img = cv2.imread(image_name1)
    r_img = cv2.imread(image_name2)
    l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)

    l_img = np.expand_dims(l_img, axis=0)
    r_img = np.expand_dims(r_img, axis=0)

    predicted_pre = test_model.predict([l_img, r_img])[0][0]

    print("predict 값", predicted_pre)
    # return round(predicted_pre,4)
    return predicted_pre
    # K.clear_session()