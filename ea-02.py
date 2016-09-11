from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation

#Generate Dataset
X_train = np.random.rand(60000,2) * 100.0 - 50.0
Y_train = X_train[:,0] + X_train[:,1]

X_test = np.random.rand(10000,2) * 100.0 - 50.0
Y_test = X_test[:,0] + X_test[:,1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(1,input_shape=(2,),init='uniform', activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')  # Using mse loss results in faster convergence

model.fit(X_train, Y_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, Y_test, batch_size=16,verbose=1)

print('Test score:',score)

print('Layer count =',len(model.layers), ' type = ',type(model.layers[0])

for k, l in enumerate(model.layers):
    weights = l.get_weights()
    print('len weights =',len(weights))
    for n, param in enumerate(weights):
    	print('param type = ',type(param), 'len param =', len(param))
    	for p in enumerate(param):
    		print('param = ', p)
