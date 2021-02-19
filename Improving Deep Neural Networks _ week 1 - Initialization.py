import numpy as np
import matplotlib.pyplot as plt

#####################
#  Helper functions #
#####################

def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
            x -- A scalar or numpy array of any size.
    Return:
            s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x
    Arguments:
            x -- A scalar or numpy array of any size.
    Return:
            s -- relu(x)
    """
    s = np.maximum(0, x)
    return s

def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss)
    Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape ()
                            b1 -- bias vector of shape ()
                            W2 -- weight matrix of shape ()
                            b2 -- bias vector of shape ()
                            W3 -- weight matrix of shape ()
                            b3 -- bias vector of shape ()
    Returns:
            loss -- the loss function (vanilla logistic loss)
    """
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache

def backward_propagation(X, Y, cache):
    """
    Implement the backward propagation
    Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            cache -- cache output from forward_propagation()
    Returns:
            gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    # Layer 3
    dz3 = 1. / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)
    # Layer 2
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    # Layer 1
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3, "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Arguments:
            parameters -- python dictionary containing your parameters
            grads -- python dictionary containing your gradients, output of n_model_backward
            learning_rate -- the learning rate, scalar
    Returns:
            parameters -- python dictionary containing your updated parameters
                          parameters['W' + str(i)] = ...
                          parameters['b' + str(i)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    # Update rule for each parameter
    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]
    return parameters

def compute_loss(a3, Y):
    """
    Implement the loss function
    Arguments:
            a3 -- post-activation, output of forward propagation
            Y -- "true" labels vector, same shape as a3
    Returns:
            loss - value of the loss function
    """
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1. / m * np.nansum(logprobs)
    return loss

##############################
#  Initialization functions  #
##############################

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
            layer_dims -- python array (list) containing the size of each layer.
    Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                            b1 -- bias vector of shape (layers_dims[1], 1)
                            ...
                            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                            bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_random(layers_dims):
    """
    Arguments:
            layer_dims -- python array (list) containing the size of each layer.
    Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                            b1 -- bias vector of shape (layers_dims[1], 1)
                            ...
                            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                            bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def initialize_parameters_he(layers_dims):
    """
    Arguments:
            layer_dims -- python array (list) containing the size of each layer.
    Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                            b1 -- bias vector of shape (layers_dims[1], 1)
                            ...
                            WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                            bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims) - 1  # integer representing the number of layers
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

###############
#  The model  #
###############

def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    """
    Implements a 3-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    Arguments:
            X -- input data, of shape (2, number of examples)
            Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
            learning_rate -- learning rate for gradient descent
            num_iterations -- number of iterations to run gradient descent
            print_cost -- if True, print the cost every 1000 iterations
            initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    Returns:
            parameters -- parameters learnt by the model
    """
    grads = {}
    costs = []  # to keep track of the loss
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)  # Compute Loss
        grads = backward_propagation(X, Y, cache)  # Backward propagation
        parameters = update_parameters(parameters, grads, learning_rate)  # Update parameters
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters