import numpy as np

def standardize_data(X:np.ndarray, y:np.ndarray):
    #if temporal, last index is time so we don't standardize it
    for j in range(X.shape[1]):
        mean,sd = np.mean(X[:,j]), np.std(X[:,j])
        X[:,j] = (X[:,j] - mean) / sd
    
    mean,sd = np.mean(y), np.std(y)
    y = (y-mean) / sd
    return X, y

def preprocess_data(data:np.ndarray, target_index:int, eta:int):
    """For input features of shape (length,dim), add lags into the dim dimension.
    Arranges data as x_1(t-eta), x_2(t-eta), ..., x_1(t-eta+1), ... x_1(t-1), ..., x_i(t)
    where i != target_index."""

    lagged = np.hstack([data[i:-eta+i] for i in range(eta)]) #lagged vars
    contemporaneous = np.delete(data[eta:],target_index,axis=1) #contemporaneous vars
    X = np.hstack((lagged,contemporaneous))
    y = data[eta:,target_index]

    X, y = standardize_data(X, y)
    
    return X,y