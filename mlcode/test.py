import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.losses import MSE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from initializer import InitFromFile

from mpi4py import MPI

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

    kf = KFold(8)

    mse_tot = 0
    rsq_tot = 0


    #8 fold cross validation
    for i, (tr_i, tst_i) in enumerate(kf.split(X)):

        model = Sequential()
        rbflayer = RBFLayer(40,
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
                batch_size=4000,
                epochs=3000,
                verbose=2)

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
        rsq_tot += r2_score(y_test, y_pred)
        mse_tot += MSE(y_test, y_pred)
        

    print(f"Average r2: {rsq_tot / 8:.5f}")
    print(f"Average MSE: {mse_tot / 8:.5f}")
    plt.savefig("blah.png")

    # saving to from file
    filename = f"rbf_{type(initializer).__name__}.h5"
    print(f"Save model to file {filename} ... ", end="")
    model.save(filename)
    print("OK")

    # save, widths & weights separately
    np.save("centers", centers)
    np.save("widths", widths)
    np.save("weights", outputlayer.get_weights()[0])


def test_init_from_file(X, y):

    print("-"*20 + " test init from file " + "-"*20)

    # load the last model from file
    filename = f"rbf_InitFromFile.h5"
    print(f"Load model from file {filename} ... ", end="")
    model = load_model(filename,
                       custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    res = model.predict(X).squeeze()  # y was (50, ), res (50, 1); why?
    print(f"MSE: {MSE(y, res):.4f}")

    # load the weights of the same model separately
    rbflayer = RBFLayer(10,
                        initializer=InitFromFile("centers.npy"),
                        betas=InitFromFile("widths.npy"),
                        input_shape=(1,))
    print("rbf layer created")
    outputlayer = Dense(1,
                        kernel_initializer=InitFromFile("weights.npy"),
                        use_bias=False)
    print("output layer created")

    model2 = Sequential()
    model2.add(rbflayer)
    model2.add(outputlayer)

    res2 = model2.predict(X).squeeze()
    print(f"MSE: {MSE(y, res2):.4f}")
    # print("Same responses: ", all(res == res2))


if __name__ == "__main__":

    X, y = load_data()

    # test simple RBF Network with random  setup of centers
    # test(X, y, InitCentersRandom(X))

    # test simple RBF Network with centers set up by k-means
    test(X, y, InitCentersKMeans(X))

    # test simple RBF Networks with centers loaded from previous
    # computation
    # test(X, y, InitFromFile("centers.npy"))

    # test InitFromFile initializer
    # test_init_from_file(X, y)
