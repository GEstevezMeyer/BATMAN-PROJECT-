import numpy as np 

def normalize_vector(vector:np.array) -> np.array:
    return (vector-np.mean(vector))/np.std(vector)

def normalize_vectors(vectors): 
    res = []
    for vector in vectors: 
        res.append(normalize_vector(vector))
    res = np.array(res)

    return res