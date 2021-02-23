import numpy as np
import math

##############################################
#  Helper function for optimizing the model  #
##############################################

###  Update parameters with gradient descent  ###

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    Arguments:
            parameters -- python dictionary containing your parameters to be updated:
                            parameters['W' + str(l)] = Wl
                            parameters['b' + str(l)] = bl
            grads -- python dictionary containing your gradients to update each parameters:
                            grads['dW' + str(l)] = dWl
                            grads['db' + str(l)] = dbl
            learning_rate -- the learning rate, scalar.
    Returns:
            parameters -- python dictionary containing your updated parameters
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    # Update rule for each parameter:
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

###  Build mini-batches from the training set  ###

def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Arguments:
            X -- input data, of shape (input size, number of examples)
            Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
            mini_batch_size -- size of the mini-batches, integer
    Returns:
            mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]  # number of training examples
    mini_batches = []
    # Step 1 - Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    # Step 2 - Partition (shuffled_X, shuffled_Y) (without the end case)
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (if last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, int(m / mini_batch_size) * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, int(m / mini_batch_size) * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

### Gradient descent with momentum  ###

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
            parameters -- python dictionary containing your parameters.
                            parameters['W' + str(l)] = Wl
                            parameters['b' + str(l)] = bl
    Returns:
            v -- python dictionary containing the current velocity.
                            v['dW' + str(l)] = velocity of dWl
                            v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    Arguments:
            parameters -- python dictionary containing your parameters:
                            parameters['W' + str(l)] = Wl
                            parameters['b' + str(l)] = bl
            grads -- python dictionary containing your gradients for each parameters:
                            grads['dW' + str(l)] = dWl
                            grads['db' + str(l)] = dbl
            v -- python dictionary containing the current velocity:
                            v['dW' + str(l)] = ...
                            v['db' + str(l)] = ...
            beta -- the momentum hyperparameter, scalar
            learning_rate -- the learning rate, scalar
    Returns:
            parameters -- python dictionary containing your updated parameters
            v -- python dictionary containing your updated velocities
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    # Momentum update for each parameter
    for l in range(L):
        # compute velocities
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads['db' + str(l + 1)]
        # update parameters
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]
    return parameters, v

###  Adam Optimization  ###

def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
            parameters -- python dictionary containing your parameters.
                            parameters["W" + str(l)] = Wl
                            parameters["b" + str(l)] = bl
    Returns:
            v -- python dictionary that will contain the exponentially weighted average of the gradient.
                            v["dW" + str(l)] = ...
                            v["db" + str(l)] = ...
            s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                            s["dW" + str(l)] = ...
                            s["db" + str(l)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}
    # Initialize v, s
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
        s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    Arguments:
            parameters -- python dictionary containing your parameters:
                            parameters['W' + str(l)] = Wl
                            parameters['b' + str(l)] = bl
            grads -- python dictionary containing your gradients for each parameters:
                            grads['dW' + str(l)] = dWl
                            grads['db' + str(l)] = dbl
            v -- Adam variable, moving average of the first gradient, python dictionary
            s -- Adam variable, moving average of the squared gradient, python dictionary
            learning_rate -- the learning rate, scalar.
            beta1 -- Exponential decay hyperparameter for the first moment estimates
            beta2 -- Exponential decay hyperparameter for the second moment estimates
            epsilon -- hyperparameter preventing division by zero in Adam updates
    Returns:
            parameters -- python dictionary containing your updated parameters
            v -- Adam variable, moving average of the first gradient, python dictionary
            s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary
    # Perform Adam update on all parameters:
    for l in range(L):
        # Exponentially weighted moving average
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]
        # bias-correction
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta1 ** t)
        # Exponentially weighted moving average of the squared gradients
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads['dW' + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads['db' + str(l + 1)])
        # bias-correction
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - beta2 ** t)
        # Update parameters;
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)
    return parameters, v, s

###########################################
#  Helper function for the network model  #
###########################################

def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
            x -- A scalar or numpy array of any size.
    Return:
            s -- sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Compute the relu of x
    Arguments:
            x -- A scalar or numpy array of any size.
    Return:
            s -- relu(x)
    """
    return np.maximum(0, x)

def initialize_parameters(layer_dims):
    """
    Arguments:
            layer_dims -- python array (list) containing the dimensions of each layer in our network
    Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            b1 -- bias vector of shape (layer_dims[l], 1)
                            Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                            bl -- bias vector of shape (1, layer_dims[l])
    """
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert (parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l - 1])
        assert (parameters['W' + str(l)].shape == layer_dims[l], 1)
    return parameters

def compute_cost(a3, Y):
    """
    Implement the cost function
    Arguments:
            a3 -- post-activation, output of forward propagation
            Y -- "true" labels vector, same shape as a3
    Returns:
            cost - value of the cost function
    """
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)
    return cost


def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the loss)
    Arguments:
            X -- input dataset, of shape (input size, number of examples)
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
    Implement the backward propagation presented in figure 2.
    Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            cache -- cache output from forward_propagation()
    Returns:
            gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
    # layer 3
    dz3 = 1. / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)
    # layer 2
    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    # layer 1
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3, "da2": da2, "dz2": dz2, "dW2": dW2,
                 "db2": db2, "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    return gradients

#######################
#  The network model  #
#######################

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.
    Arguments:
            X -- input data, of shape (2, number of examples)
            Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
            layers_dims -- python list, containing the size of each layer
            optimizer -- string indicating the optimizer used:
                            'gd' - Mini-batch Gradient descent
                            'momentum' - Mini-batch gradient descent with momentum
                            'adam' - Mini-batch with Adam mode
            learning_rate -- the learning rate, scalar.
            mini_batch_size -- the size of a mini batch
            beta -- Momentum hyperparameter
            beta1 -- Exponential decay hyperparameter for the past gradients estimates
            beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
            epsilon -- hyperparameter preventing division by zero in Adam updates
            num_epochs -- number of epochs
            print_cost -- True to print the cost every 1000 epochs
    Returns:
            parameters -- python dictionary containing your updated parameters
    """
    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    m = X.shape[1]  # number of training examples
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)
    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    # Optimization loop
    for i in range(num_epochs):
        # Define the random mini-batches
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0
        for minibatch in minibatches:
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)
            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)
            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)
    return parameters
