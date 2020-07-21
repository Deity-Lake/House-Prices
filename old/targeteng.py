import numpy as np

def target_eng(data, target_column, inverse = False):
    
    if(inverse):
        y = np.exp(data)
    else:
        y = np.log(np.ravel(data[[target_column]]))   
    
    return y