import numpy as np

def get_data1(nsample=1000):
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1)))
    y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)
    return x_data, y_data

def get_data2(nsample=1000):
    tmp = get_data1(nsample)
    return (tmp[1], tmp[0])
