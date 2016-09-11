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
model.add(Dense(20,activation='tanh'))
model.add(Dense(1,activation='tanh'))
model.compile(loss='mae', optimizer='rmsprop')  # Using mse loss results in faster convergence

def SetWeights(chromo):
	w = chromo.genes[0,0:1]; w.shape = (1,1);
	b = chromo.genes[0,1:1+1]; b.shape = (1,)
	model.layers[0].set_weights([w,b])#set weights and biases
	w = chromo.genes[0,1+1:1+1+20]; w.shape = (1,20)
	b = chromo.genes[0,1+1+20:1+1+20+20]; b.shape = (20,)
	model.layers[1].set_weights([w,b])
	w = chromo.genes[0,1+1+20+20:1+1+20+20+20]; w.shape= (20,1)
	b = chromo.genes[0,1+1+20+20+20:1+1+20+20+20+1]; b.shape = (1,)
	model.layers[2].set_weights([w,b])


#################################
#Evolutionary Algo for optimizing Neural Netowrks
import numpy as np
import sys
import datetime
import math

#EA Parameters
gene_count = 63
population_size = 100

p_mutation = 0.15 #1.0/gene_count
p_crossover = 0.5
#mutation factor is amount by which the original chromosome gets
#changed by after applying the mutate decison mask vector

mutation_factor = 2.0 #Weights are mutated by a value in the range of +/- mutation_factor/2
max_gen = 2000

nb_eval_batch = len(X_train) / 3
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

	    #self.genes  = np.zeros((1,gene_count)) #np.random.rand(1,gene_count) * 1.0 - 0.50
	    self.genes  = np.random.rand(1,gene_count) * 2.0 - 1.0
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
	begin = 0; end = 1;
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
	begin = 1; end = 1+1;
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
	#layer 3
	begin = 1+1; end = 1+1+20;
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
	begin = 1+1+20; end = 1+1+20+20;
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
	#layer 6
	begin = 1+1+20+20; end = 1+1+20+20+20;
	MutatePart(winner,loser,p_mutation,p_crossover,begin,end)
	begin = 1+1+20+20+20; end = 1+1+20+20+20+1;
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
score_ = model.evaluate(X_test,Y_test, batch_size=nb_batch_size,verbose=0)
print('Test score:',score_)

print('time taken =',(datetime.datetime.utcnow() - t1))
print('time taken to set gene =',total_gene_set_time)

results = model.predict(X_test,batch_size=nb_batch_size)
if 1:
	fig = plt.figure()

	ax1 = fig.add_subplot(2,1,1)
	ax1.plot(X_train,Y_train,'k*',label='Training Set')
	ax1.plot(X_test,Y_test,'r.',label='Test Set')
	ax1.plot(X_test,results,'g+',label='Prediction')
	ax1.set_xlabel('input'); ax1.set_ylabel('output')
	ax1.set_title('Training/Test data & Prediction')
	ax1.legend(loc='best')

	ax2 = fig.add_subplot(2,1,2)
	ax2.plot(fitlog_gen, fitlog_loss, 'r-')
	ax2.set_xlabel('generation');ax2.set_ylabel('MAE')
	ax2.set_title('Training Performance')

	fig.savefig('lossplot-{}.png'.format(sys.argv[1]))
	plt.close(fig)