#Bibliotecas utilizadas na rede neural técnica
import numpy as np
import math
import pandas as pd
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}\\..\\teste.csv", sep=';')
    #Changing pandas dataframe to numpy array
    X = df.iloc[:, 2:7].values #colunas 1-7
    Y = df.iloc[:, 7:8].values #coluna 8 de saídas esperadas
    #Normalizing the data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1)
    
    np.random.seed(2)
    
    #No. of neurons in first layer
    n_x = 5
    #No. of neurons in hidden layer
    n_h = 12
    #No. of neurons in output layer
    n_y = 1

    num_of_epochs = 10
    learning_rate = 0.3    

    parameters = model(X_train, Y_train, n_x, n_h, n_y, num_of_epochs, learning_rate)
    y_predict = predict(X_test, Y_test, parameters)

    # outboundfile = open(f"{os.path.dirname(os.path.abspath(__file__))}\\..\\output.txt")
    # outboundfile.write(y_predict)
    # outboundfile.close()

#função ativação
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_x, n_h)
    b1 = np.zeros((1, n_h))
    W2 = np.random.randn(n_h, n_y)
    b2 = np.zeros((n_y, 1))

    parameters = {
            "W1": W1,
            "b1" : b1,
            "W2": W2,
            "b2" : b2
            }
    return parameters

def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.matmul(X, W1) + b1
    A1 = np.tanh(Z1) #função ativação da primeira camada
    Z2 = np.matmul(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    cache = {
      "A1": A1,
      "A2": A2
    }
    return A2, cache


def calculate_cost(A2, Y):
    cost = np.multiply([1 - y for y in Y], np.log([1 - a2 for a2 in A2])) - np.multiply(Y, np.log(A2))
    cost = np.squeeze(cost)
    return cost


def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    W2 = parameters["W2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1).T
    db2 = dZ2
    dZ1 = np.multiply(np.dot(W2, dZ2).T, [1 - x for x in np.power(A1, 2)])
    dW1 = np.dot(np.array(X).T, dZ1)
    db1 = dZ1
    
    grads = {
      "dW1": dW1,
      "db1": db1,
      "dW2": dW2,
      "db2": db2
    }
    
    return grads


def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
      
    new_parameters = {
      "W1": W1,
      "W2": W2,
      "b1": b1,
      "b2": b2
    }
    return new_parameters


def model(X, Y, n_x, n_h, n_y, num_of_epochs, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)
    loss = {}
    for j in range(len(X)):
        train = []
        for i in range(num_of_epochs):
            entry = [X[j]]
            output = [Y[j]]
            a2, cache = forward_prop(entry, parameters)
            cost_train = calculate_cost(a2, output)
            train.append(math.fabs(cost_train.item()))
            grads = backward_prop(entry, output, cache, parameters)
            parameters = update_parameters(parameters, grads, learning_rate)
        loss[j] = train
    loss_series = pd.DataFrame(loss).transpose().mean()
    loss_df = pd.DataFrame(loss_series, columns=["Loss"])
    loss_df.to_csv("loss_per_epoch.csv")
    plt.plot(loss_df)
    plt.title(f'Loss')
    plt.ylabel('Loss')
    plt.legend(['Epoch'], loc='upper left')
    plt.savefig("loss_per_epoch.png")

    return parameters


def predict(X, Y, parameters):
    loss = {}
    prev = {}
    for i in range(len(X)):
        a2, cache = forward_prop([X[i]], parameters)
        yhat = np.squeeze(a2)
        cost_pred = math.fabs(calculate_cost(a2, [Y[i]]))
        loss[i] = [cost_pred]
        if yhat >= 0.7:
          y_predict = 1
        elif yhat < 0.7 and yhat >0.3:
          y_predict = 0.5
        else:
          y_predict = 0
        prev[i] = [y_predict]
    loss = pd.DataFrame(loss)
    prev = pd.DataFrame(prev)
    prev.to_csv("prev_test.csv")
    loss.to_csv("loss_test.csv")
    plt.plot(pd.DataFrame(loss))
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.legend(['Test'], loc='upper left')
    plt.show()
    
    return prev

if __name__ == '__main__':
    main()