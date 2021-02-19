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

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5, 4))
    parameters["b1"] = theta[20:25].reshape((5, 1))
    parameters["W2"] = theta[25:40].reshape((3, 5))
    parameters["b2"] = theta[40:43].reshape((3, 1))
    parameters["W3"] = theta[43:46].reshape((1, 3))
    parameters["b3"] = theta[46:47].reshape((1, 1))
    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    return theta

#######################
#  1D Gradient Check  #
#######################

def forward_propagation(x, theta):
    """
    Arguments:
            x -- a real-valued input
            theta -- our parameter, real number
    Returns:
        J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    J = theta * x
    return J

def backward_propagation(x, theta):
    """
    Computes the derivative of J with respect to theta
    Arguments:
            x -- a real-valued input
            theta -- our parameter, real number
    Returns:
            dtheta -- the gradient of the cost with respect to theta
    """
    dtheta = x
    return dtheta

def gradient_check(x, theta, epsilon=1e-7):
    """
    Arguments:
            x -- a real-valued input
            theta -- our parameter, a real number as well
            epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    Returns:
            difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # step 1 - compute 'gradapprox'
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus)
    J_minus = forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)
    # step 2 - compute 'gradapprox' using backward_propagation
    grad = backward_propagation(x, theta)
    # step 3 - compute the relative difference between 'grad' and 'gradapprox'
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    return difference

########################
#  N-D Gradient Check  #
########################

def forward_propagation_n(X, Y, parameters):
    """
    Arguments:
            X -- training set for m examples
            Y -- labels for m examples
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape (5, 4)
                            b1 -- bias vector of shape (5, 1)
                            W2 -- weight matrix of shape (3, 5)
                            b2 -- bias vector of shape (3, 1)
                            W3 -- weight matrix of shape (1, 3)
                            b3 -- bias vector of shape (1, 1)
    Returns:
            cost -- the cost function (logistic cost for one example)
    """
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    # Cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache

def backward_propagation_n(X, Y, cache):
    """
    NOTE:
        This function deliberately has a mistake!
    Arguments:
            X -- input datapoint, of shape (input size, 1)
            Y -- true "label"
            cache -- cache output from forward_propagation_n()
    Returns:
            gradients -- A dictionary with the gradients of the cost with respect to each parameter,
                         activation and pre-activation variables.
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    # layer 3
    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    # layer 2
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    # layer 1
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2, "dZ2": dZ2, "dW2": dW2,
                 "db2": db2, "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    Arguments:
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
            grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
            x -- input datapoint, of shape (input size, 1)
            y -- true "label"
            epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    Returns:
            difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    # Compute gradapprox
    for i in range(num_parameters):
        # J_plus[i]
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))
        # Compute J_minus[i]
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    return difference
