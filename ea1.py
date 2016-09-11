#Evolutionary Algo for optimizing Neural Netowrks
from __future__ import print_function
import numpy as np
import sys
import datetime
import math

#EA Parameters
gene_count = 10#669706
population_size = 20

p_mutation = 1.0/gene_count
p_crossover = 0.5

loss_delta = 1
avg_loss_prev = 0;

class Chromosome:
	"""docstring for Chromosome"""
	fitness = 0.0
	is_fitness_invalid = True # used for computing fitness 
	def __init__(self, gene_count):
		self.gene_count = gene_count
		self.genes  = np.random.rand(1,gene_count) * 200.0 - 100.0
		self.is_fitness_invalid = True

def ComputeFitness(pop,min_loss):
	"""
		Computes fitness each chromosome, 
		returns avgloss, min_loss and min_loss_index
	"""
	total_fitness = 0.0
	min_loss_index = -1
	
	for i in range(0,pop.size):
		#print i,pop[0,i]
		if pop[0,i].is_fitness_invalid:
			#TODO: 
			# 1. set the gene to the NN topology
			# 2. evaluate against the whole *Training* dataset
			# 3. resulting 'TestScore' will be the fitness
			#SetWeights(pop[0,i])
			#pop[0,i].fitness = EvalModel()
	
			#Mock fitness computation
			pop[0,i].fitness = pop[0,i].genes.mean(axis=1)
			print(i,' computed fitness')

			pop[0,i].is_fitness_invalid = False
			if min_loss >= pop[0,i].fitness:
				min_loss = pop[0,i].fitness
				min_loss_index = i
				print('min-loss = ',min_loss,'i = ',i)
		total_fitness = total_fitness + pop[0,i].fitness

	return (total_fitness / pop.size, min_loss, min_loss_index)


def Mutate(winner,loser,p_mutation,p_crossover):
	if np.random.rand() < p_crossover:
		#generate crossover site
		cs = math.floor(np.random.rand() * (winner.gene_count-1))
		loser.genes[0,cs:] = winner.genes[0,cs:]

	#mutation factor is amount by which the original chromosome gets
	#changed by after applying the mutate decison mask vector
	mutation_factor = 10 #Weights are mutated by a value in the range of +5 to -5
	#mutate prep
	m1 = np.random.rand(1,loser.gene_count) #mutation decision probability vector
	mask = m1 < p_mutation; #decision as a boolean mask
	#vector of mutations
	m2 = np.random.rand(1,loser.gene_count) * mutation_factor - (mutation_factor/2)
	mutation = mask * m2 # vector of mutation to be added
	loser.genes = loser.genes + mutation
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
while loss_delta > 0.001:
	(avg_loss, min_loss, mi) = ComputeFitness(population,min_loss)
	if mi >= 0:
		best_so_far = population[0,mi]
		print('best_so_far changed mi = ',mi)
	loss_delta = math.fabs(avg_loss - avg_loss_prev)
	avg_loss_prev = avg_loss
	print('[{}] [{}] best-so-far = {} mi = {} min-loss = {} loss_delta = {}'.format(\
					 str(datetime.datetime.utcnow()), \
					 generation_count, \
					 best_so_far.fitness,\
					 mi,\
					 min_loss, \
					 loss_delta))
	

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
			print(I2,' mutated')
		else:		
			#P2 is better, so we replace P1
			population[0,I1] = Mutate(P2,P1,p_mutation,p_crossover)
			print(I1,' mutated')

	generation_count += 1

print('time taken =',(datetime.datetime.utcnow() - t1))
