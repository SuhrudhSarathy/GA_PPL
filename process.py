'''File that contains the basics of this algorithm'''
import random 
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np

def make_choice(x, y):
	if x!= 0 and y!= 0:
		choice = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
		if choice == 1:
			x, y = x+1, y+1
		elif choice == 2:
			x, y = x+1, y
		elif choice == 3:
			x, y = x+1, y-1
		elif choice == 4:
			x, y = x-1, y+1
		elif choice == 5:
			x, y = x-1, y
		elif choice == 6:
			x, y = x-1, y-1
		elif choice == 7:
			x, y = x, y+1
		elif choice == 8:
			x, y = x, y-1
	else :
		choice = random.choice([1, 2, 3])
		if choice == 1:
			x, y = x+1, y+1
		elif choice == 2:
			x, y = x+1, y
		elif choice == 3:
			x, y = x, y+1
	return x, y
def sigmoid(x):
	sigmoid = 1/(1+np.exp(-x))
	return sigmoid
def fitness_finder(bot):
	return bot.fitness
def vector_is_positive(vector):
	if vector[0] < 0 and vector[1] < 0:
		state = False
	elif vector[0] < 0 and vector[1] > 0:
		state = True

class Population():
	"""Contains the population. Here we will perfrom activites for the population"""
	def __init__(self, popsize, initial, final):
		self.popsize = popsize
		self.initial = initial
		self.final = final
		self.population = []
		self.best_ind = None
		self.best_fitness = 0
	def make_pop(self):
		for i in range(0, self.popsize):
			ind = Individual((self.initial[0], self.initial[1]), (self.final[0], self.final[1]))
			ind.make_individual()
			ind.check_status()
			ind.fitness_calc()
			self.population.append(ind)
	def sort_fitness(self):
		for ind in self.population:
			ind.fitness_calc()
		self.population = sorted(self.population, key=fitness_finder, reverse=True)
	def get_best(self):
		self.best_ind = self.population[0]
		self.best_fitness = self.population[0].fitness
		self.population[0].plot_individual()
	def start(self):
		self.make_pop()
		self.sort_fitness()
		self.get_best()


class Individual():
	"""Class for the individuals that folow gentic algorithm and then reproduce themselves"""
	def __init__(self, initial, final):
		self.initial = initial
		self.final = final
		self.chromosome = [self.initial]
		self.x = self.initial[0]
		self.y = self.initial[1]
		self.X = self.final[0]
		self.Y = self.final[1]
		self.length = 0
		self.distance = 0
		self.fitness = 0
		self.status = False
		self.turns = 0
		self.final_reward = 0
	def make_individual(self):
		while self.x != self.X and self.Y!=self.y:
			self.x = make_choice(self.x, self.y)[0]
			self.y = make_choice(self.x, self.y)[1]
			self.chromosome.append((self.x, self.y))
	def check_status(self):
		if self.X == self.chromosome[-1][0] and self.Y == self.chromosome[-1][1]:
			self.status = True
			self.final_reward = 50
	def fitness_calc(self):
		for i in range(len(self.chromosome)-1):
			x = self.chromosome[i][0]
			y = self.chromosome[i][1]
			X = self.chromosome[i+1][0]
			Y = self.chromosome[i+1][1]
			self.length += sqrt((X-x)**2 + (Y-y)**2)
			vector = ((X-x), (Y-y))
			if not vector_is_positive(vector):
				self.turns += 1
		self.distance = sqrt((self.X-self.chromosome[-1][0])**2 + (self.Y - self.chromosome[-1][1])**2)
		self.fitness = (1/(75*self.distance + self.length))  - (10**-5)*self.turns 


	def plot_individual(self):
		x_points = []
		y_points = []
		for obj in self.chromosome:
			x_points.append(obj[0])
			y_points.append(obj[1])
		plt.scatter([self.X], [self.Y], color="green")
		plt.scatter(x_points, y_points, color = "red", s=2)
		plt.plot(x_points, y_points, color=[0.6, 0.6, 0.6, 0.6])
		plt.show()
	
if __name__ == '__main__':
	initial = (0, 0)
	final = (10, 10)
	population = Population(10, initial, final)
	population.start()
	print(population.best_fitness, population.population[0].status)



	
			
			



