#Bibliotecas utilizadas na rede neural técnica
import numpy as np
import pandas as pd
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():

    df = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}\\..\\teste.csv", sep=';')
    #Changing pandas dataframe to numpy array
    X = df.iloc[:,2:7].values #colunas 2-7
    Y = df.iloc[:,8:8].values #coluna 8 de saídas esperadas
    print(Y) #debug
    #Normalizing the data
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1)
    
    np.random.seed(2)
    
    #No. of neurons in first layer
    n_x = 5
    #No. of neurons in hidden layer
    n_h = 12
    #No. of neurons in output layer
    n_y = 1
    
    num_of_iters = 20000
    learning_rate = 0.3    

    parameters = model(X_train, Y_train, n_x, n_h, n_y, num_of_iters, learning_rate)
    print(parameters) #debug
    y_predict = predict(X_test, Y_test, parameters)
    print(y_predict) #debug

    outboundfile = open(f"{os.path.dirname(os.path.abspath(__file__))}\\..\\output.txt")
    outboundfile.write(y_predict)
    outboundfile.close()

#função ativação
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
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

def calculate_cost(A2, Y, X):
    cost = -np.sum(np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2)))/X.shape[1]
    cost = np.squeeze(cost)
    
    return cost

def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    W2 = parameters["W2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True)/X.shape[1]
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True)/X.shape[1]
    
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
      "b1" : b1,
      "b2" : b2
    }
    
    return new_parameters

def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters = initialize_parameters(n_x, n_h, n_y)
    print(parameters)
    train = list()
    for i in range(0, num_of_iters+1):
      a2, cache = forward_prop(X, parameters)
      cost_train = calculate_cost(a2, Y, X)
      train.append(cost_train, a2)
      grads = backward_prop(X, Y, cache, parameters)
      parameters = update_parameters(parameters, grads, learning_rate)
    plt.plot(train)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.legend(['Train'], loc='upper left') 
    plt.show()
    
    return parameters

def predict(X, parameters,Y):
    pred = list()
    a2, cache = forward_prop(X, parameters)
    yhat = np.squeeze(a2)
    cost_pred = calculate_cost(a2,Y, X)
    pred.append(cost_pred, a2)
    if(yhat >= 0.7):
      y_predict = 1
    elif (yhat < 0.7 and yhat >0.3):
      y_predict = 0.5
    else:
      y_predict = 0
          
    plt.plot(pred)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.legend(['Test'], loc='upper left')
    plt.show()
    
    return y_predict

if __name__ == '__main__':
    main()