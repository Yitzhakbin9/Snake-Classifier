import numpy as np
from matplotlib import pyplot as plt


##############################################
#               For NN model                 #
##############################################
# Make predictions using the calculated weights and biases
def predictNN(x,w1,b1,w2,b2):
    a1 = np.dot(w1,x)+b1
    z1 = np.maximum(0,a1)
    a2 = np.dot(w2,z1)+b2
    probs = 1/(1+np.exp(-a2))
    y_pred = (probs > 0.5).astype(int)
    return y_pred

# Compute the percentage match between predicted and target values
def accuracy(label,y_pred):
    acc = np.mean(np.equal(label,y_pred))
    return acc

def showErrorGraphNN(params_train , costs_train , costs_dev):
    print("Dev error {:5.4f} on epoch {}.".format(params_train["cost_dev"], params_train["epoch"]))
    train_curve, = plt.plot(costs_train, label='Train error')
    test_curve, = plt.plot(costs_dev, label='Dev error')
    plt.legend(handles=[train_curve, test_curve])
    plt.xlabel('Epochs')
    plt.ylabel('Mean log loss')
    plt.show()


##############################################
#           For Logistic Regression          #
##############################################

def predict(x,w,b):
    z = np.dot(w,x)+b
    probs = 1/(1+np.exp(-z))
    y_pred = (probs > 0.5).astype(int)
    return y_pred

def showErrorGraph(costs_train , costs_dev):
    train_curve, = plt.plot(costs_train, label = 'Train error')
    test_curve,  = plt.plot(costs_dev, label = 'Dev error')
    plt.legend(handles=[train_curve,test_curve])
    plt.xlabel('Epochs')
    plt.ylabel('Mean log loss')
    plt.show()