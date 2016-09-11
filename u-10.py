'''Train a simple deep NN on the MNIST dataset.
Get to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
import sys
import datetime

nb_batch_size = 128
nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))


rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)
#-----------

def SetWeights(chromo):
	#There are 8 layers we move in one layer at a time and set the weights
	w = chromo.genes[0,0:401408]; w.shape = (784, 512);
	b = chromo.genes[0,401408:401408+512]; b.shape = (512,)
	model.layers[0].set_weights([w,b])#set weights and biases
	w = chromo.genes[0,401408+512:401408+512+262144]; w.shape = (512, 512)
	b = chromo.genes[0,401408+512+262144:401408+512+262144+512]; b.shape = (512,)
	model.layers[3].set_weights([w,b])
	w = chromo.genes[0,401408+512+262144+512:401408+512+262144+512+5120]; w.shape= (512, 10)
	b = chromo.genes[0,401408+512+262144+512+5120:401408+512+262144+512+5120+10]; b.shape = (10,)
	model.layers[6].set_weights([w,b])


#################################
#Evolutionary Algo for optimizing Neural Netowrks
import numpy as np
import sys
import datetime
import math

#EA Parameters
gene_count = 669706
population_size = 30

p_mutation = 0.15 #1.0/gene_count
p_crossover = 0.5
#mutation factor is amount by which the original chromosome gets
#changed by after applying the mutate decison mask vector

mutation_factor = 2.0 #Weights are mutated by a value in the range of +/- mutation_factor/2
max_gen = 20

nb_eval_batch = len(X_train) / 100
loss_delta = 1
avg_loss_prev = 0;
total_gene_set_time = datetime.datetime.utcnow() - datetime.datetime.utcnow()


def EvalModel(i):
	#score = model.evaluate(X_train, Y_train, show_accuracy=True, verbose=0)
	#print('score:', score[0],'accuracy:', score[1])
	#return score

	# nb_batch = 100
	# p = len(X_train)/nb_batch - 1
	# i = int(math.floor(np.random.rand() * p + 0.5) * nb_batch)
	
	tr_batch = X_train[i:i+nb_eval_batch,:]
	label_batch = Y_train[i:i+nb_eval_batch]
	score = model.evaluate(tr_batch, label_batch, batch_size=nb_batch_size,verbose=0)
	#score = model.evaluate(X_train, Y_train, batch_size=nb_batch_size,verbose=0)
	
	#print('[',i,'] score:', score[0],'accuracy:', score[1])
	return score

class Chromosome:
	"""docstring for Chromosome"""
	fitness = 0.0
	accuracy = 0.0
	is_fitness_invalid = True # used for computing fitness
	def __init__(self, gene_count):
	    self.gene_count = gene_count

	    self.genes  = np.zeros((1,gene_count)) #np.random.rand(1,gene_count) * 1.0 - 0.50
	    #self.genes  = np.random.rand(1,gene_count) * 2.0 - 1.0
	    #GetWeights(self)
	    #self.genes  = np.random.rand(1,gene_count) * 1.0 - 0.50
	    self.is_fitness_invalid = True

def ComputeFitness(pop,min_loss):
	"""
	Computes fitness each chromosome,
	returns avgloss, min_loss and min_loss_index
	"""
	total_fitness = 0.0
	min_loss_index = -1
	#####
	p = len(X_train)/nb_eval_batch - 1
	j = int(math.floor(np.random.rand() * p + 0.5) * nb_eval_batch)
	#####
	global total_gene_set_time
	for i in range(0,pop.size):
		if pop[0,i].is_fitness_invalid:
			# 1. set the gene to the NN topology
			# 2. evaluate against the whole *Training* dataset
			# 3. resulting 'TestScore' will be the fitness
			t2 = datetime.datetime.utcnow()
			SetWeights(pop[0,i])
			total_gene_set_time += datetime.datetime.utcnow() - t2
			res = EvalModel(j)
			pop[0,i].fitness = res #loss
			#pop[0,i].accuracy  = res[1]#accuracy
			#Mock fitness computation
			#pop[0,i].fitness = pop[0,i].genes.mean(axis=1)
			#print(i,' computed fitness')

			pop[0,i].is_fitness_invalid = False
		if min_loss >= pop[0,i].fitness:
			min_loss = pop[0,i].fitness
			min_loss_index = i
		total_fitness = total_fitness + pop[0,i].fitness

	return (total_fitness / pop.size, min_loss, min_loss_index)

def MutatePart(winner,loser,p_mutation,p_crossover,begin,end):
	count = end - begin
	#if np.random.rand() < p_crossover:
	#	#generate crossover site
	#	cs = math.floor(np.random.rand() * (count-1))
	#	loser.genes[0,begin:end] = winner.genes[0,begin:end]

	#multi point crossover
	m0 = np.random.rand(1,count) #crossover decision probability vector
	mask = m0 < p_crossover #decision as a boolean mask
	#vector of crossovers
	crossover = (mask * winner.genes[0,begin:end]) + (~mask * loser.genes[0,begin:end])
	loser.genes[0,begin:end] = crossover

	#mutate prep
	m1 = np.random.rand(1,count) #mutation decision probability vector
	mask = m1 < p_mutation; #decision as a boolean mask
	#vector of mutations
	m2 = np.random.rand(1,count) * mutation_factor - (mutation_factor/2)
	mutation = mask * m2 # vector of mutation to be added
	loser.genes[0,begin:end] = loser.genes[0,begin:end] + mutation

def Mutate(winner,loser,p_mutation,p_crossover):
    #apply mutation and cross over layer by layer 
    #layer 0
    begin = 0; end = 401408;
    MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
    begin = 401408; end = 401408+512;
    MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
    #layer 3
    begin = 401408+512; end = 401408+512+262144;
    MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
    begin = 401408+512+262144; end = 401408+512+262144+512;
    MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
    #layer 6
    begin = 401408+512+262144+512; end = 401408+512+262144+512+5120;
    MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
    begin = 401408+512+262144+512+5120; end = 401408+512+262144+512+5120+10;
    MutatePart(winner,loser,p_mutation,p_crossover,begin,end)

    loser.is_fitness_invalid = True
    return loser

#-------------------------------------------------------------------------------------------------
#initialize population
print('Starting ', datetime.datetime.utcnow())
vChromosome = np.vectorize(Chromosome)#vectorize Chromosome constructor
arg_array = np.full((1,population_size),gene_count,dtype=int)#init array with gene_count as value
population = vChromosome(arg_array)#create a population of Chromosomes

#aim is to minimize the loss
t1 = datetime.datetime.utcnow() # for timing
min_loss = sys.maxint
best_so_far = None
generation_count = 0
mi_prev = -1
same_mi_counter = 0
fitlog_loss = np.zeros(max_gen)
fitlog_gen = np.zeros(max_gen)

while generation_count < max_gen: #loss_delta > 0.001:
	(avg_loss, min_loss, mi) = ComputeFitness(population,min_loss)
	if mi >= 0:
	    best_so_far = population[0,mi]
	loss_delta = avg_loss - avg_loss_prev
	avg_loss_prev = avg_loss
	# print('[{}] [{}] best-so-far = {} mi = {} min-loss = {} loss_delta = {}'.format(\
	#                                 str(datetime.datetime.utcnow()), \
	#                                 generation_count, \
	#                                 best_so_far.fitness,\
	#                                 mi,\
	#                                 min_loss, \
	#                                 loss_delta))

	fitlog_gen[generation_count] = generation_count
	fitlog_loss[generation_count] = min_loss

	# if mi_prev == mi:
	# 	same_mi_counter +=1
	# else:
	# 	mi_prev = mi
	# 	same_mi_counter = 0

	# if same_mi_counter > 50:
	# 	p_mutation = p_mutation + 0.05
	# 	same_mi_counter = 0
	# 	print('p_mutation =',p_mutation)

	#prep for crossover and mutation
	idx = np.random.permutation(population_size)

	for kk in range(0,population_size/2):
	    I1 = idx[2*kk]
	    I2 = idx[2*kk+1]
	    P1 = population[0,I1]
	    P2 = population[0,I2]

	    #minimization, so <=
	    if P1.fitness <= P2.fitness:
	        #P1 is better, so we replace P2
	        population[0,I2] = Mutate(P1,P2,p_mutation,p_crossover)
	    else:
	        #P2 is better, so we replace P1
	        population[0,I1] = Mutate(P2,P1,p_mutation,p_crossover)

	generation_count += 1



#one last time
(avg_loss, min_loss, mi) = ComputeFitness(population,min_loss)
if mi >= 0:
    best_so_far = population[0,mi]
loss_delta = avg_loss - avg_loss_prev
avg_loss_prev = avg_loss
print('*[{}] [{}] best-so-far = {} mi = {} min-loss = {} loss_delta = {}'.format(\
                                str(datetime.datetime.utcnow()), \
                                generation_count, \
                                best_so_far.fitness,\
                                mi,\
                                min_loss, \
                                loss_delta))

print('==========================================')
#evaluate the 'best so far' on test set
SetWeights(best_so_far)

score_ = model.evaluate(X_test,Y_test, show_accuracy=True, verbose=0)
print('Test score:',score_[0],'Test accuracy:',score_[1])

print('time taken =',(datetime.datetime.utcnow() - t1))
print('time taken to set gene =',total_gene_set_time)

results = model.predict(X_test,batch_size=nb_batch_size)
if 1:
	fig = plt.figure()

	ax2 = fig.add_subplot(1,1,1)
	ax2.plot(fitlog_gen, fitlog_loss, 'r-')
	ax2.set_xlabel('generation');ax2.set_ylabel('MSE')
	ax2.set_title('Training Performance')

	fig.savefig('lossplot-{}.png'.format(sys.argv[1]))
	plt.close(fig)