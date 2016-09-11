# Simple single neuron network to model a regression task
from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

#Generate Dataset
X_train = np.random.rand(600,2) * 100.0 - 50.0
Y_train = X_train[:,0] + X_train[:,1]

X_test = np.random.rand(100,2) * 100.0 - 50.0
Y_test = X_test[:,0] + X_test[:,1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(1,input_shape=(2,),init='uniform', activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')  # Using mse loss results in faster convergence

def GetWeights(chromo):
	[w,b] = model.layers[0].get_weights() #get weights and biases
	w.shape =(401408,)
	b.shape = (512,)
	chromo.genes[0,0:401408] = w
	chromo.genes[0,401408:401408+512] = b

	[w,b] = model.layers[3].get_weights()
	w.shape = (262144,)
	b.shape = (512,)
	chromo.genes[0,401408+512:401408+512+262144] = w
	chromo.genes[0,401408+512+262144:401408+512+262144+512] = b

	[w,b] = model.layers[6].get_weights()
	w.shape = (5120,)
	b.shape = (10,)
	chromo.genes[0,401408+512+262144+512:401408+512+262144+512+5120] = w
	chromo.genes[0,401408+512+262144+512+5120:401408+512+262144+512+5120+10] = b

def SetWeights(chromo):
	#There are 8 layers we move in one layer at a time and set the weights
	w = chromo.genes[0,0:2]; w.shape = (2,1);
	b = chromo.genes[0,2:3]; b.shape = (1,)
	model.layers[0].set_weights([w,b])#set weights and biases

def EvalModel(j):
	# nb_batch = 16
	# p = len(X_train)/nb_batch - 1
	# i = int(math.floor(np.random.rand() * p + 0.5) * nb_batch)
	# tr_batch = X_train[i:i+nb_batch,:]
	# label_batch = Y_train[i:i+nb_batch]
	# score = model.evaluate(tr_batch, label_batch, batch_size=nb_batch,verbose=0)
	score = model.evaluate(X_train, Y_train,verbose=0)
	#print('[',j,'] Eval score:',score)
	return score

######################

#Evolutionary Algo for optimizing Neural Netowrks
import numpy as np
import sys
import datetime
import math

#EA Parameters
gene_count = 3
population_size = 100

p_mutation = 0.15 #1.0/gene_count
p_crossover = 0.5 #0.0001

loss_delta = 1
avg_loss_prev = 0;
total_gene_set_time = datetime.datetime.utcnow() - datetime.datetime.utcnow()
class Chromosome:
	"""docstring for Chromosome"""
	fitness = 0.0
	is_fitness_invalid = True # used for computing fitness
	def __init__(self, gene_count):
	    self.gene_count = gene_count
	    self.genes  = np.random.rand(1,gene_count) * 2.0 - 1.0
	    #GetWeights(self)
	    self.is_fitness_invalid = True

def ComputeFitness(pop,min_loss):
	"""
	Computes fitness each chromosome,
	returns avgloss, min_loss and min_loss_index
	"""
	total_fitness = 0.0
	min_loss_index = -1
	global total_gene_set_time
	for i in range(0,pop.size):
		if pop[0,i].is_fitness_invalid:
			# 1. set the gene to the NN topology
			# 2. evaluate against the whole *Training* dataset
			# 3. resulting 'TestScore' will be the fitness
			t2 = datetime.datetime.utcnow()
			SetWeights(pop[0,i])
			total_gene_set_time += datetime.datetime.utcnow() - t2
			pop[0,i].fitness = EvalModel(i)

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
	if np.random.rand() < p_crossover:
		#generate crossover site
		cs = math.floor(np.random.rand() * (count-1))
		loser.genes[0,begin:end] = winner.genes[0,begin:end]

	#mutation factor is amount by which the original chromosome gets
	#changed by after applying the mutate decison mask vector
	mutation_factor = 2.0 #Weights are mutated by a value in the range of +/- mutation_factor/2
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
	begin = 0; end = 2;

	#for c in enumerate(loser.genes):
	#	print('c = ',c)
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
	
	#for c in enumerate(loser.genes):
	#	print('c = ',c)
	#print('++++++++++++++++++++++++++')
	
	#print('-----')	
	#for k, l in enumerate(model.layers):
	#    weights = l.get_weights()
	#    print('len weights =',len(weights))
	#    for n, param in enumerate(weights):
	#    	for p in enumerate(param):
	#    		print('param = ', p)

	begin = 2; end = 3
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)

	loser.is_fitness_invalid = True
	return loser

#-------------------------------------------------------------------------------------------------
#initialize population
vChromosome = np.vectorize(Chromosome)#vectorize Chromosome constructor
arg_array = np.full((1,population_size),gene_count,dtype=int)#init array with gene_count as value
population = vChromosome(arg_array)#create a population of Chromosomes

#aim is to minimize the loss
t1 = datetime.datetime.utcnow() # for timing
min_loss = sys.maxint
best_so_far = None
generation_count = 0;
while generation_count < 1000: #loss_delta > 0.001:
	(avg_loss, min_loss, mi) = ComputeFitness(population,min_loss)
	if mi >= 0:
	    best_so_far = population[0,mi]
	loss_delta = math.fabs(avg_loss - avg_loss_prev)
	avg_loss_prev = avg_loss
	#print('[{}] [{}] best-so-far = {} mi = {} min-loss = {} loss_delta = {}'.format(\
	#                                 str(datetime.datetime.utcnow()), \
	#                                 generation_count, \
	#                                 best_so_far.fitness,\
	#                                 mi,\
	#                                 min_loss, \
	#                                 loss_delta))

	#prep for crossover and mutation
	idx = np.random.permutation(population_size)

	for kk in range(0,population_size/2):
	    I1 = idx[2*kk]
	    I2 = idx[2*kk+1]
	    P1 = population[0,I1]
	    P2 = population[0,I2]

	    #print('I1 =',I1,'I2 =',I2,'P2 fitness =',P1.fitness,'P2 fitness =',P2.fitness)
	    #minimization, so <=
	    if P1.fitness <= P2.fitness:
	        #P1 is better, so we replace P2
	        population[0,I2] = Mutate(P1,P2,p_mutation,p_crossover)
	    else:
	        #P2 is better, so we replace P1
	        population[0,I1] = Mutate(P2,P1,p_mutation,p_crossover)

	generation_count += 1

print('==========================================')
#evaluate the 'best so far' on test set
SetWeights(best_so_far)
score_ = model.evaluate(X_test,Y_test, verbose=1)
print('Test score:',score_)

print('time taken =',(datetime.datetime.utcnow() - t1))
print('time taken to set gene =',total_gene_set_time)
for k, l in enumerate(model.layers):
    weights = l.get_weights()
    print('len weights =',len(weights))
    for n, param in enumerate(weights):
    	for p in enumerate(param):
    		print('param = ', p)

