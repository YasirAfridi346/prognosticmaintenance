import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import webbrowser
import time
from tensorflow import keras 
from numpy import genfromtxt
import math
from sklearn.metrics import mean_squared_error

def make_prediction_csv(filename):
    path = 'static/'+ filename
    data = genfromtxt(path, delimiter=',')
    data=pd.DataFrame(data)
    scaler=MinMaxScaler(feature_range=(-1,1))
    model_testing=scaler.fit_transform(data)
    seq_size=5
    x_test,y_test=seq(model_testing,seq_size)
    n_features=1
    x_test= x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    #from keras.models import load_model
    model=keras.models.load_model('model/MyLSTM_Model.h5')
    test_predict=model.predict(x_test)
    test_predict1=scaler.inverse_transform(test_predict)
    y_test1=scaler.inverse_transform([y_test])
    y_test1=y_test1.T
    Next_prediction_size=int(len(test_predict1)*0.1)
    Next_prediction_size
    predicted_x_test,predicted_y_test=Prerdict_future(test_predict1,Next_prediction_size)
    n_features=1
    predicted_x_test= predicted_x_test.reshape((predicted_x_test.shape[0], predicted_x_test.shape[1], n_features))
    next_test_predict=model.predict(predicted_x_test)
    test_predict2=scaler.inverse_transform(next_test_predict)
    y_test2=scaler.inverse_transform([predicted_y_test])
    y_test2=y_test2.T
    test = np.zeros((len(test_predict1),1), dtype=int)
    test =np.concatenate((test, test_predict2))
    plt.plot(test, color='b', linestyle='--', label='Future 10% Predicted')
    plt.plot(test_predict1, color='r', linestyle='--', label='Predicted Values')
    plt.plot(y_test1, color='g', alpha=0.2, label='Actual Values Values')
    plt.legend()
    plt.savefig('static/images/prediction.png')
    test_score=math.sqrt(mean_squared_error(y_test,test_predict))
    test_score
    return test_score



def seq(dataset,seq_size=5):
    x=[]
    y=[]
    for i in range(len(dataset)-seq_size-1):
        window=dataset[i:i+seq_size,0]
        x.append(window)
        y.append(dataset[i+seq_size,0])
    return np.array(x) , np.array(y)


def Prerdict_future(prediction,Next_prediction_size):
    seq_size=5
    x=[]
    y=[]
    for i in range(len(prediction)-Next_prediction_size,len(prediction)):
        if (i+seq_size +5)< len(prediction):
            window=prediction[i:i+seq_size,0]
            x.append(window)
            y.append(prediction[i+seq_size,0])
    return np.array(x) , np.array(y)

def allowed_file(filename):
    """
    Checks if a given file `filename` is of type image with 'png', 'jpg', or 'jpeg' extensions
    """
    allowed_extensions = {'csv', '.xls'}
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in allowed_extensions)