import numpy as np

# CE(y,yˆ) =  -SUMyi log(y^-i)
def compute_cost(predictions, labels):
    return -np.sum(np.multiply(labels, np.log(predictions)))
