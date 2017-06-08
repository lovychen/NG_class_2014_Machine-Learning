import matplotlib.pyplot as plt
from numpy import *
def showdata(train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 0], train_x[i, 1], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 0], train_x[i, 1], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    # weights = weights.getA()  # convert mat to array
    # y_min_x = float(-weights[0] - weights[1] * min_x)/weights[2]
    # y_max_x = float(-weights[0] - weights[1] * max_x)/weights[2]
    # plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()