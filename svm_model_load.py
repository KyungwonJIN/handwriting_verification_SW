from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import pickle
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## -4~-2.5 정도로 single predict threshold 잡고 하면 될듯?
## .pkl로 저장된 svm model을 불러와서 예측
def multimodal(predict_value):
    # file_name =
    if len(predict_value) == 2:
        clf_from_joblib = joblib.load('./svm_model_save/predict_two.pkl')
        y_pred = clf_from_joblib.predict([predict_value])
    elif len(predict_value) == 3:
        clf_from_joblib = joblib.load('./svm_model_save/predict_three.pkl')
        y_pred = clf_from_joblib.predict([predict_value])
    elif len(predict_value) == 4:
        clf_from_joblib = joblib.load('./svm_model_save/predict_four.pkl')
        y_pred = clf_from_joblib.predict([predict_value])
    else:
        y_pred = [0]
    return y_pred[0]