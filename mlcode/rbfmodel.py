import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.losses import MSE
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

import matplotlib.pyplot as plt

from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from initializer import InitFromFile
import time

def load_data():

    data = np.loadtxt("../FinalData.csv")
    X = data[:, :-1]  # except last column
    y = data[:, -1]  # last column only
    return X, y

    


def test(X, y, initializer):

    title = f" test {type(initializer).__name__} "
    print("-"*20 + title + "-"*20)

    plt.figure(dpi=1200)
    figure, axis = plt.subplots(2, 4)

    kf = KFold(8, shuffle=True, random_state=42)

    mse_tot = 0
    rsq_tot = 0


    #8 fold cross validation
    for i, (tr_i, tst_i) in enumerate(kf.split(X)):

        model = Sequential()
        rbflayer = RBFLayer(10,
                            initializer=initializer,
                            betas=1,
                            input_shape=(40,))
        outputlayer = Dense(1, use_bias=False)

        model.add(rbflayer)

        model.add(outputlayer)

        model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

        X_train = X[tr_i]
        X_test = X[tst_i]
        y_train = y[tr_i]
        y_test = y[tst_i]

        model.fit(X_train, y_train,
                batch_size=10000,
                epochs=2000)

        y_pred = model.predict(X_test)


        x = np.linspace(-0.1,1,100)
        y_plot = x

        # show graph
        axis[int(i/4), i%4].scatter(y_test, y_pred)  # prediction
        axis[int(i/4), i%4].set(xlabel = "Actual value", ylabel = "Predicted Value")
        axis[int(i/4), i%4].plot(x, y_plot, '-r')
        axis[int(i/4), i%4].plot([-0.1, 1], [0, 0], color='black')  # zero line

        # plot centers
        centers = rbflayer.get_weights()[0]
        widths = rbflayer.get_weights()[1]
    

        # calculate and print MSE
        y_pred = y_pred.squeeze()
        
        r2 = r2_score(y_test, y_pred)
        rsq_tot += r2
        mse = MSE(y_test, y_pred)
        mse_tot += mse

        print(r2)
        print(mse)


        np.save("centers", centers)
        np.save("widths", widths)
        np.save("weights", outputlayer.get_weights()[0])

        filename = f"rbf_{type(initializer).__name__}.h5.{i}"
        print(f"Save model to file {filename} ... ", end="")
        model.save(filename)
        print("OK")
        break

    print(f"Average r2: {rsq_tot / 8:.5f}")
    print(f"Average MSE: {mse_tot / 8:.5f}")
    plt.savefig("final.png")
    



def test_init_from_file(X, y):

    print("-"*20 + " test init from file " + "-"*20)

    # load the last model from file
    filename = f"rbf_InitCentersKMeans.h5.5"
    print(f"Load model from file {filename} ... ", end="")
    model = load_model(filename,
                       custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    plot_model(model, show_shapes=True, dpi=1200)
    # exit()

    kf = KFold(8)

    for i, (tr_i, tst_i) in enumerate(kf.split(X)):

        y_test = y[tst_i]
        X_test = X[tst_i]
        y_train = y[tr_i]
        x_train = X[tr_i]

        start = time.time()
        y_pred = model.predict(X_test).squeeze()  
        print(time.time() - start)
        print(f"MSE: {MSE(y_test, y_pred):.4f}")
        print(f"r2: {r2_score(y_test, y_pred):.4f}")
        
        plt.scatter(y_test, y_pred, alpha = 0.002, color="deepskyblue")  # prediction
        # plt.set(xlabel = "Actual value", ylabel = "Predicted Value")
        x = np.linspace(-0.1,1,100)
        y_plot = x
        plt.plot(x, y_plot, "black")
        plt.plot([-0.1, 1], [0, 0], color='black')  # zero line
        plt.plot([0,0], [-0.1,1], color = "black")
        plt.title("Actual Hourly Solar Yield V Model's Prediction")
        plt.xlabel("Actual Value")
        plt.ylabel("Predicted Value")
        plt.show()


if __name__ == "__main__":

    X, y = load_data()


    # test simple RBF Network with centers set up by k-means
    test(X, y, InitCentersKMeans(X))

    # test InitFromFile initializer
    test_init_from_file(X, y)
