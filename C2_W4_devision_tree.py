

# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    p = y.sum()/y.shape[0]
    if (p != 0) and (p != 1):
        entropy = entropy + ( -p * np.log2(p) - (1-p)*np.log2(1-p) )
        
    ### END CODE HERE ###        
    
    return entropy





# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    left_indices = []
    right_indices = []
    
    ### START CODE HERE ###
    for i in node_indices:
        
        if X[i,feature] == 1:
            left_indices.append(i)
        else :
            right_indices.append(i)
        
            
        
            
    ### END CODE HERE ###
        
    return left_indices, right_indices



# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    ### START CODE HERE ###
    w_left = y_left.shape[0] / (y_left.shape[0] + y_right.shape[0])
    w_right = y_right.shape[0] / (y_left.shape[0] + y_right.shape[0])
    if w_left == 0:
        return 0.0
    if w_right == 0:
        return 0.0
    information_gain = compute_entropy( y_node ) - (w_left * compute_entropy( y_left ) +  w_right * compute_entropy( y_right ) )
    ### END CODE HERE ###  
    
    return information_gain





# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    
    ### START CODE HERE ###
    max = 0
    s = (X.shape[1])
    for i in range (s):
        gain = compute_information_gain( X , y , node_indices, i)
        if( gain > max ):
            best_feature = i
            max = gain

    ### END CODE HERE ##    
   
    return best_feature