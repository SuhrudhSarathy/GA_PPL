#!/usr/bin/env python3

from shapely.geometry import Polygon, LineString, Point
import random
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KDTree
import math

MAP_X = 10
MAP_Y = 10
X = list(np.arange(0.5, MAP_X, 0.5))
Y = list(np.arange(0.5, MAP_Y, 0.5))

class Obstacle():
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.coords = [(self.x+0.5, self.y+0.5),(self.x+0.5, self.y-0.5), (self.x-0.5, self.y-0.5),(self.x-0.5, self.y+0.5)]
        self.polygon = Polygon(self.coords)

    def plot_obstacle(self):
        x = [point[0] for point in self.coords]
        y = [point[1] for point in self.coords]
        plt.fill(x, y, color = 'black')


class Node():
    def __init__(self, x, y):
        '''
            Node parameters:
                1. Coordinate (self.coordinate)
                2. First Safety Level (self.fss)
                3. Second Safety Level (self.sss)
        '''
        self.x, self.y = x, y
        self.coordinate = Point(self.x, self.y)
        self.fss_coords = [(self.x+1, self.y+1),(self.x+1, self.y-1), (self.x-1, self.y-1),(self.x-1, self.y+1),(self.x+1, self.y+1)]
        self.sss_coords = [(self.x+2, self.y+2),(self.x+2, self.y-2), (self.x-2, self.y-2),(self.x-2, self.y+2), (self.x+2, self.y+2)]
        self.fss = LineString(self.fss_coords)
        self.sss = LineString(self.sss_coords)
        self.fss_polygon = Polygon(self.fss_coords)
        self.sss_polygon = Polygon(self.sss_coords)
    
    def check_safety(self, obstacle):
        '''
            Returns (Bool, Bool) corrsponding to conditions
        ''' 
        obstacle = obstacle.polygon
        if self.fss.intersects(obstacle) or obstacle.within(self.fss_polygon):
            fss = True
        else:
            fss = False
        
        if self.sss.intersects(obstacle) or obstacle.within(self.sss_polygon):
            sss = True
        else: 
            sss = False

        return fss, sss

    def plot_node_params(self):
        x_fss, y_fss = [point[0] for point in self.fss.coords], [point[1] for point in self.fss.coords]
        x_sss, y_sss = [point[0] for point in self.sss.coords], [point[1] for point in self.sss.coords]
        plt.plot(x_fss, y_fss, color = 'red')
        plt.plot(x_sss, y_sss, 'r--')
        plt.scatter(self.x, self.y, color='red')


class Map():
    def __init__(self, n_obst):
        self.n_obst = n_obst
        self.nodes =[]
        self.obstacles = []
        self.start = Point(0, 0)
        self.goal = Point(10, 10)
        self.node_points = []
        self.tree = None

    def init_obst(self):
        for i in range(self.n_obst):
            obstacle = Obstacle(random.choice(X), random.choice(Y))
            self.obstacles.append(obstacle)

    def init_nodes(self):
        for i in range(MAP_X + 1):
            for j in range(MAP_Y + 1):
                node = Node(i, j)
                self.nodes.append(node)
                for obstacle in self.obstacles:
                    if Point(i, j).within(obstacle.polygon) or obstacle.polygon.intersects(Point(i, j)):
                        self.nodes.remove(node)
                        break
                    else:
                        continue
        for node in self.nodes:
            self.node_points.append([node.coordinate.x, node.coordinate.y])
        self.tree = KDTree(np.array(self.node_points), leaf_size=2) #KDTree - useful for finding nearest neighbors
                       
    def plot_map(self):
        for obstacle in self.obstacles:
            obstacle.plot_obstacle()
        plt.scatter([node.x for node in self.nodes], [node.y for node in self.nodes], color='red')
        plt.show()
    
    def init_map(self):
        self.init_obst()
        self.init_nodes()


class Individual():
    def __init__(self, env):
        self.map = env
        self.path = [] # chromosome will be a sequence of points that make up the path
        self.fitness = 0 # will be automatically updated by the fitness function
        self.fscost = 0 # First safety cost 
        self.sscost = 0 # second safety cost
        self.rotation_cost = 0 # no. of rotations made (consider if rotation angle greater than 90*)
        self.length = 0 # the length used is calculated using euclidean distance
        self.goal_cost = 0
        self.reached_goal = False # will be updated if individual reaches goal based on conditions

    def init_chromosome(self):
        '''
            Makes a chromosome
        '''
        self.path = [Node(0, 0)]
        current_node = self.path[0]
        current_x, current_y = current_node.x, current_node.y
        while True:            
            dist, ind = self.map.tree.query(np.array([[current_x, current_y]]), 6)
            nn = random.choice(ind[0]) # converting nparray to list
            if self.check_intersection((current_x, current_y), (self.map.node_points[nn][0], self.map.node_points[nn][1])):
                continue
            else:
                current_node= self.map.nodes[nn]
                self.path.append(current_node)
                current_x, current_y = current_node.x, current_node.y
                if current_x == self.map.goal.x and current_y == self.map.goal.y:
                    self.reached_goal = True
                    break
            if len(self.path) > 2000 and self.reached_goal == False:
                self.goal_cost = 1
                break

        self.path_line = LineString([(point.x, point.y) for point in self.path])
        '''print(self.path[-1].coordinate, len(self.path))
        plt.plot([p[0] for p in self.path_line.coords], [p[1] for p in self.path_line.coords], color='green')'''
        

    def check_intersection(self, point1, point2):
        line = LineString([point1, point2])
        for obstacle in self.map.obstacles:
            if line.intersects(obstacle.polygon):
                collision = True
                break
            else:
                collision = False
        return collision
    
    def get_all_costs(self):
        for i in range(len(self.path)):
            for obstacle in self.map.obstacles:
                if self.path[i].check_safety(obstacle)[0]:
                    self.fscost += 1
                if self.path[i].check_safety(obstacle)[1]:
                    self.sscost += 1
        try:
            if math.atan((self.path[i].x - self.path[i+1].x)/(self.path[i].y - self.path[i+1].y)) > np.pi/2 or math.atan((self.path[i].x - self.path[i+1].x)/(self.path[i].y - self.path[i+1].y)) < -np.pi/2:
                self.rotation_cost += 1
                self.length += np.sqrt((self.path[i].x - self.path[i+1].x)**2 + (self.path[i].y - self.path[i+1].y)**2)
        except ZeroDivisionError :
            self.rotation_cost += 1
        except IndexError:
            pass                     
    
    def get_fitness(self, wl, ws1, ws2, wgc):
        self.fitness = (1 / (wl * self.length + ws1 * self.fscost + ws2 * self.sscost)) - self.rotation_cost - wgc * self.goal_cost

    def make_new_ind(self):
        self.init_chromosome()
        self.get_all_costs()
        self.get_fitness(0.09, 0.005, 0.001, 0.5)
        print('Done')

    def make_all_zero(self):
        self.path = [] # chromosome will be a sequence of points that make up the path
        self.fitness = 0 # will be automatically updated by the fitness function
        self.fscost = 0 # First safety cost 
        self.sscost = 0 # second safety cost
        self.rotation_cost = 0 # no. of rotations made (consider if rotation angle greater than 90*)
        self.length = 0 # the length used is calculated using euclidean distance

    def plot_ind(self):
        self.path_line = LineString([(point.coordinate.x, point.coordinate.y) for point in self.path])
        plt.plot([p[0] for p in self.path_line.coords], [p[1] for p in self.path_line.coords], color='green')


class Population():
    def __init__(self, pop_size, env):
        self.map = env
        self.pop_size = pop_size
        self.population = []
        self.parents_elite = []
        self.parents_ts = []
        self.children = []

    def make_pop(self):
        for i in range(self.pop_size):
            ind = Individual(self.map)
            ind.make_new_ind()
            self.population.append(ind)

    def selection(self):
        print('selection')
        # using Elitist Selection and Truncation Selection
        self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)

        #Using Elitism to select the top individuals as it is
        selection_probability = 0.2 * len(self.population)
        for i in range(int(selection_probability)):
            self.parents_elite.append(self.population[i])
        
        #selecting using Truncation Selection (TRS)
        p = random.randrange(30, 50)/100
        selection_pressure = len(self.population) * p
        for i in range(int(selection_pressure + 1)):
            parent = random.choice(self.population[int(selection_probability):])
            self.parents_ts.append(parent)        
    
    def crossover(self):
        childlist = []
        crosspointlist = []
        mathpointlist = []
        # Approach as proposed in the paper (SAME ADJACENCY CROSSOVER)
        p1 = random.choice(self.parents_elite)
        p2 = random.choice(self.parents_ts)
        vf = len(p1.path)
        vs = len(p2.path)

        #Start main loop
        for i in range(vf):
            print('crossover')
            for j in range(vs):
                try:
                    if self.isFeasible(p1.path[i+1], p2.path[j]) == True and self.isFeasible(p2.path[j+1], p1.path[i]) == True:
                        crosspointlist.append(i)
                        mathpointlist.append(j)
                except IndexError:
                    pass
        for k in range(len(crosspointlist)):
            print('crossover2')
            #getting variables i, j
            i = crosspointlist[k]
            j = mathpointlist[k]
            
            try:
            #Offspring Generation
                offspring1 = Individual(my_map)
                offspring2 = Individual(my_map)

                # making offspring2
                
                offspring1.path.extend(p1.path[:i+1])
                offspring1.path.extend(p2.path[j+1:])

                #making offspring1
                offspring2.path.extend(p2.path[:j+1])
                offspring2.path.extend(p1.path[i+1:])

                #Add both offspring to childlists
                childlist.append(offspring1)
                childlist.append(offspring2)

            except IndexError:
                pass

        childlist = sorted(childlist, key=lambda individual: individual.fitness)
        self.children.append(childlist[0])
        self.children.append(childlist[1])

    def isFeasible(self, parent1, parent2):
        '''
            Feasibility Conditions:
            1. Path joining two points should not pass through an obstacle
            2. Path length <= sqrt(2)
        '''
        point1 = parent1.coordinate
        point2 = parent2.coordinate
        line = LineString([point1, point2])
        for obstacle in self.map.obstacles:
            if line.intersects(obstacle.polygon) or np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2) <= np.sqrt(2):
                isFeasible = False
                break
            else:
                isFeasible = True
        return isFeasible

    def mutation(self, individual):
        mutation_node = random.choice(individual.path[1:-1])
        dist, ind = self.map.tree.query(np.array([[mutation_node.coordinate.x, mutation_node.coordinate.y]]), 3)
        nn = random.choice(ind[0])
        change_node = self.map.nodes[nn]
        if self.isFeasible(mutation_node, change_node) == True:
            index = individual.path[1:-1].index(mutation_node)
            individual.path.remove(mutation_node)
            individual.path.insert(index, change_node)

        else:
            pass
        
    def mutate_population(self):    
        print('mutation')    
        prob = random.random()
        for i in range(len(self.children)):
            if prob < 0.2:
                print('Mutation')
                self.mutation(self.children[i])

    def set_to_zero(self):
        self.parents_elite = []
        self.parents_ts = []
        self.children = []

    def start_evolution(self, max_iter):
        iter_num = 0
        self.make_pop()
        while iter_num < max_iter:
            self.selection()
            self.crossover()
            self.mutate_population()
            self.children = self.population
            self.set_to_zero()
            print(iter_num + 1)
            iter_num += 1 
        self.fittest = self.get_fittest()

    def get_fittest(self):
        fittest = self.population[0] 
        return fittest   

if __name__ == '__main__':
    my_map = Map(5)
    my_map.init_map()
    population = Population(100, my_map)
    population.start_evolution(100)
    
    


