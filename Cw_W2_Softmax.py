# UNQ_C1
# GRADED CELL: my_softmax

def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    ### START CODE HERE ### 
    
    a = np.zeros(z.shape)
    myExp = np.exp(z)
    mySum = myExp.sum()
    for i in range(z.shape[0]):
        a[i] = myExp[i]/mySum

    ### END CODE HERE ### 
    return a
	
	
	
	# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        ### START CODE HERE ### 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(10, activation = 'linear')
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)


	