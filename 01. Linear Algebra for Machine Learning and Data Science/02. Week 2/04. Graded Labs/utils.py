from sympy import symbols, Eq, Matrix, solve, sympify
import numpy as np

def string_to_augmented_matrix(equations):
    # Split the input string into individual equations
    equation_list = equations.split('\n')
    equation_list = [x for x in equation_list if x != '']
    # Create a list to store the coefficients and constants
    coefficients = []
    
    ss = ''
    for c in equations:
        if c in 'abcdefghijklmnopqrstuvwxyz':
            if c not in ss:
                ss += c + ' '
    ss = ss[:-1]
    
    # Create symbols for variables x, y, z, etc.
    variables = symbols(ss)
    # Parse each equation and extract coefficients and constants
    for equation in equation_list:
        # Remove spaces and split into left and right sides
        sides = equation.replace(' ', '').split('=')
        
        # Parse the left side using SymPy's parser
        left_side = sympify(sides[0])
        
        # Extract coefficients for variables
        coefficients.append([left_side.coeff(variable) for variable in variables])
        
        # Append the constant term
        coefficients[-1].append(int(sides[1]))

    # Create a matrix from the coefficients
    augmented_matrix = Matrix(coefficients)
    augmented_matrix = np.array(augmented_matrix).astype("float64")

    A, B = augmented_matrix[:,:-1], augmented_matrix[:,-1].reshape(-1,1)
    
    return ss, A, B

