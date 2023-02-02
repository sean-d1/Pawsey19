import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import MSE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score
import keras
import keras.layers as layers

import matplotlib.pyplot as plt

from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from initializer import InitFromFile

from sklearn.model_selection import StratifiedKFold

def load_data(data = "../FinalData.csv"):

    data = np.loadtxt(data)
    X = data[:, :-1]  # except last column
    y = data[:, -1]  # last column only
    return X, y

def get_input(X, y):


    skf = StratifiedKFold(10, shuffle=True, random_state=42)

    filename = f"rbf_InitCentersKMeans.h5.5"
    print(f"Load model from file {filename} ... ")
    rbf = load_model(filename,
                        custom_objects={'RBFLayer': RBFLayer})

    r2 = 0
    mse = 0

    size = [x[0] for x in X]

    groups = [int(i * 10) for i in size]

    for i, (tr_i, tst_i) in enumerate(skf.split(X, groups)):

        print(i)

        x_train = X[tr_i]
        x_test = X[tst_i]
        y_train = y[tr_i]
        y_test = y[tst_i]
        
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        x_train, x_test, y_train, y_test = train_test_split(X, y)

        model = Sequential()
        
        model.add(Dense(4, input_shape=(1,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(40, activation='relu'))

        model.compile(optimizer='adam', loss='mse')
        plot_model(model, show_shapes = True, dpi=1200)
        model.fit(y_train, x_train,
                    epochs=20,
                    batch_size=200)
        
        x_pred = model.predict(y_test)

        for i in range(0, len(x_pred)):
            max_hour = x_pred[i][4:28].max()
            max_month = x_pred[i][28:].max()

            done = False
            for j in range(4, 28):
                if x_pred[i][j] == max_hour and not done:
                    x_pred[i][j] = 1
                    done = True
                else:
                    x_pred[i][j] = 0
            
            done = False
            for j in range(28, 40):
                if x_pred[i][j] == max_month and not done:
                    x_pred[i][j] = 1
                    done = True
                else:
                    x_pred[i][j] = 0


        y_pred = rbf.predict(x_pred)

        print(r2_score(y_pred, y_test))
        r2 += r2_score(y_pred, y_test)

        y_pred = y_pred.squeeze()
        

        print(f"{MSE(y_pred, y_test):.4f}")
        mse += MSE(y_pred, y_test)

        plt.scatter(y_pred, y_test, alpha=0.002)
        x = np.linspace(-0.1,1,100)
        y_plot = x
        plt.plot(x, y_plot, '-r')
        plt.title("Actual Hourly Solar Yield V Prediction based on Model's Predicted Input")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.show()


    print(r2/10.0)   
    print(mse/10.0)                      

        

if __name__ == "__main__":

    X, y = load_data()
    get_input(X, y)