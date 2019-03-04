import numpy as np
import time
import math
from utils import calc_cost, time_left
from branch_and_bound import nearest_neighbor
from my_approx import christofides

def simulated_annealing(graph, time_limit, seed=None):
    start = time.time()
    
    if seed:
        np.random.seed(seed)

    min_cost = float('inf')
    opt_path = []
    solutions = []
    iteration_time = 0
    #choices = list(graph.nodes)
    approx_paths = christofides(graph, 30, method='all')
    i = 0
    #count = 0
    
    while time_left(time_limit, start) > iteration_time:# and i < len(graph) and count < 5:
        temperature = 1
        final_temperature = 0.1
        cooling_factor = 0.99
        #np.random.shuffle(curr_path)
        tabu = []
        #hoice = np.random.choice(choices, 1)[0]
        #choices.remove(choice)
        #curr_path, curr_cost = nearest_neighbor(graph, choice)
        curr_path, curr_cost = approx_paths[i]
        i += 1

        if curr_cost < min_cost:
            min_cost = curr_cost
            opt_path = curr_path
            solutions.append((opt_path, min_cost, time.time() - start))
            #print(solutions[-1])
        curr_path.pop()
        
        #print('starting path')
        #print(curr_path)
        #print('-----------------------------------------------------------------')
        
        while temperature > final_temperature:# and time_left(time_limit, start) > 10:
            for _ in range(1000):
                idxs = sorted(np.random.choice(curr_path, 2, replace=False))
                if idxs not in tabu[-len(graph):]:
                    tabu.append(idxs)
                    new_path = list(curr_path)
                    #print(new_path)
                    #print(idxs)
                    new_path[idxs[0]-1], new_path[idxs[1]-1] = new_path[idxs[1]-1], new_path[idxs[0]-1]
                    new_cost = calc_cost(graph, new_path)
                    new_cost += graph[new_path[-1]][new_path[0]]['weight']
                    diff = ((curr_cost - new_cost) / curr_cost) * 100# - curr_cost
                    #if math.exp(diff/temperature) == 0 and count==0:
                        #print(time.time() - start)
                        #count +=1
                    if diff > 0:
                        #print('accept')
                        #print(diff)
                        #print(curr_path)
                        #print(curr_cost)
                        #print(new_path)
                        #print(new_cost)
                        curr_path = new_path
                        curr_cost = new_cost
                        if curr_cost < min_cost:
                            min_cost = curr_cost
                            opt_path = curr_path + [curr_path[0]]
                            solutions.append((opt_path, min_cost, time.time() - start))
                            #print(solutions[-1])
                        #print('--------------------------')
                    elif math.exp(diff/temperature) >= np.random.random():
                        #print('accept with prob')
                        #print(diff)
                        #print(temperature)
                        #print(math.exp(diff/temperature))
                        #print('--------------------------')
                        curr_path = new_path
                        curr_cost = new_cost

            temperature *= cooling_factor
        iteration_time = time.time() - start - iteration_time
        #print(iteration_time)
        #if curr_cost >= min_cost:
        #    count += 1
        #print(count)

    #solutions.append((opt_path, min_cost, time.time() - start))
return solutions