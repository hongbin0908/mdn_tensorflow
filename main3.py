import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import data_util
import numpy as np
from model_mdn import ModelMdn

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)


x_data, y_data = data_util.get_data2(2500)

modelMdn = ModelMdn()
modelMdn.fit(x_data, y_data)

plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, modelMdn.NEPOCH,1), modelMdn.loss[100:], 'r-')
plt.savefig(os.path.join(local_path, "mdn_fig3_loss.png"))


x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

y_test = modelMdn.predict(x_test)

plt.figure(figsize=(9, 9))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.savefig(os.path.join(local_path, "mdn_fig3.png"))
