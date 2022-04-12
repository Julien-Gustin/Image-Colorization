import numpy as np

#datas a list of vector of same size 
def create_histograms(datas):
    bins = np.arange(-100, 100, 5)
    #a for us : scikit image [-127, 128]
    #b 