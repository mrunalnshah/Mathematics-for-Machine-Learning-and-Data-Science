import matplotlib.pyplot as plt
import numpy as np 

def plot_transformation(T, e1, e2):
    color_original = "#129cab"
    color_transformed = "#cc8933"
    
    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))
    
    plt.axis([-5, 5, -5, 5])
    plt.quiver([0, 0],[0, 0], [e1[0], e2[0]], [e1[1], e2[1]], color=color_original, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, e2[0][0], e1[0][0], e1[0][0]], 
             [0, e2[1][0], e2[1][0], e1[1][0]], 
             color=color_original)
    e1_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e1)])
    ax.text(e1[0]-0.2+e1_sgn[0], e1[1]-0.2+e1_sgn[1], f'$e_1$', fontsize=14, color=color_original)
    e2_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e2)])
    ax.text(e2[0]-0.2+e2_sgn[0], e2[1]-0.2+e2_sgn[1], f'$e_2$', fontsize=14, color=color_original)
    
    e1_transformed = T(e1)
    e2_transformed = T(e2)
    
    plt.quiver([0, 0],[0, 0], [e1_transformed[0], e2_transformed[0]], [e1_transformed[1], e2_transformed[1]], 
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0,e2_transformed[0][0], e1_transformed[0][0]+e2_transformed[0][0], e1_transformed[0][0]], 
             [0,e2_transformed[1][0], e1_transformed[1][0]+e2_transformed[1][0], e1_transformed[1][0]], 
             color=color_transformed)
    e1_transformed_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e1_transformed)])
    ax.text(e1_transformed[0][0]-0.2+e1_transformed_sgn[0][0], e1_transformed[1][0]-e1_transformed_sgn[1][0], 
            f'$T(e_1)$', fontsize=14, color=color_transformed)
    e2_transformed_sgn = 0.4 * np.array([[1] if i==0 else [i[0]] for i in np.sign(e2_transformed)])
    ax.text(e2_transformed[0][0]-0.2+e2_transformed_sgn[0][0], e2_transformed[1][0]-e2_transformed_sgn[1][0], 
            f'$T(e_2)$', fontsize=14, color=color_transformed)
    
    plt.gca().set_aspect("equal")
    plt.show()

def initialize_parameters(n_x):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """
    
    ### START CODE HERE ### (~ 2 lines of code)
    W = np.random.randn(1, n_x) * 0.01 # @REPLACE EQUALS None
    b = np.zeros((1, 1)) # @REPLACE EQUALS None
    ### END CODE HERE ###
    
    assert (W.shape == (1, n_x))
    assert (b.shape == (1, 1))
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost


def backward_propagation(A, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    A -- the output of the neural network of shape (1, number of examples)
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # Backward propagation: calculate dW, db.
    dZ = A - Y
    dW = 1/m * np.matmul(dZ, X.T)
    db = 1/m * np.sum(dZ, axis = 1, keepdims = True)
    
    grads = {"dW": dW,
             "db": db}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Retrieve each gradient from the dictionary "grads".
    dW = grads["dW"]
    db = grads["db"]
    
    # Update rule for each parameter.
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {"W": W,
                  "b": b}
    
    return parameters

def train_nn(parameters, A, X, Y, learning_rate = 0.01):
    # Backpropagation. Inputs: "A, X, Y". Outputs: "grads".
    grads = backward_propagation(A, X, Y)
    
    # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
    parameters = update_parameters(parameters, grads, learning_rate)
    
    return parameters
