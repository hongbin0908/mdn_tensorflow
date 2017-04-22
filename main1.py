import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data_util
import numpy as np
from model1 import Model1

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)


x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1)

x_data, y_data = data_util.get_data1()
model = Model1()
model.fit(x_data, y_data)
y_test = model.predict(x_test)

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.savefig(os.path.join(local_path, "mdn_fig1.png"))
