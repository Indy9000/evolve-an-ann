from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
import sys
import datetime

#Generate Dataset
nb_train_size = 1600
nb_test_size = 400
nb_epoch_count = 10000
nb_batch_size = 16
X_train = np.random.rand(nb_train_size,1) * 2.0 - 1.0 # input
Y_train = np.sin(X_train * 2.0 * np.pi) + np.random.normal(0, 0.1, (nb_train_size,1)) # label with noise
Y_train = Y_train * 0.5 + 0.5 #normalized to [0,1]

X_test = np.random.rand(nb_test_size,1) * 2.0 - 1.0
Y_test = np.sin(X_test * 2.0 * np.pi) + np.random.normal(0, 0.1, (nb_test_size,1)) # target with noise
Y_test = Y_test * 0.5 + 0.5 #normalized to [0,1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(1,input_shape=(1,),init='zero', activation='tanh'))
#model.add(Dense(20,activation='tanh',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
model.add(Dense(20,activation='tanh'))
#model.add(Dense(8,activation='tanh'))
model.add(Dense(1,activation='tanh'))
model.compile(loss='mse', optimizer='rmsprop')  # Using mse loss results in faster convergence

t1 = datetime.datetime.utcnow() # for timing
fitlog = model.fit(X_train, Y_train, nb_epoch=nb_epoch_count, batch_size=nb_batch_size,verbose=0)
print('training time =',(datetime.datetime.utcnow() - t1))
score = model.evaluate(X_test, Y_test, batch_size=nb_batch_size,verbose=0)

print('Test score:',score)

print('Layer count =',len(model.layers), ' type = ',type(model.layers[0]))

results = model.predict(X_test,batch_size=nb_batch_size)
if 1:
	fig = plt.figure()
	#ax.plot(fitlog.epoch, fitlog.history['acc'], 'g-')
	#ax.plot(fitlog.epoch, fitlog.history['val_acc'], 'g--')
	#ax.plot(fitlog.epoch, fitlog.history['val_loss'], 'r--')
	ax1 = fig.add_subplot(2,1,1)
	ax1.plot(X_train,Y_train,'k*',label='Training Set')
	ax1.plot(X_test,Y_test,'r.',label='Test Set')
	ax1.plot(X_test,results,'g+',label='Prediction')
	ax1.set_xlabel('input'); ax1.set_ylabel('output')
	ax1.set_title('Training/Test data & Prediction')
	ax1.legend(loc='best')

	ax2 = fig.add_subplot(2,1,2)
	ax2.plot(fitlog.epoch, fitlog.history['loss'], 'r-')
	ax2.set_xlabel('epoch');ax2.set_ylabel('MSE')
	ax2.set_title('Training Performance')

	fig.savefig('lossplot-{}.png'.format(sys.argv[1]))
	plt.close(fig)

# for k, l in enumerate(model.layers):
# 	weights = l.get_weights()
# 	print('len weights =',len(weights))
# 	for n, param in enumerate(weights):
# 		print('param type = ',type(param), 'len param =', len(param))
# 		for p in enumerate(param):
# 			print('param = ', p)

##############################################