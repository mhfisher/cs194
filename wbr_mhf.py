import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import helpers
import wbr_sim

def draw_result(result):
  """ Draw simulation result"""
  nx.draw_kamada_kawai(nx.Graph(result))
  plt.show()

num_players = 100
strategies = [helpers.PA_strategy]
strategy_choices = [0 for i in range(num_players)]

high_alpha = lambda x, y: 0.75
result_high_alpha = wbr_sim.run_simulation(num_players, strategies, strategy_choices,
                                high_alpha, random_walk=False)
draw_result(result_high_alpha)


low_alpha = lambda x, y: 0.25
result_low_alpha = wbr_sim.run_simulation(num_players, strategies, strategy_choices,
                                low_alpha, random_walk=False)
# draw_result(result_low_alpha)


uniform = [helpers.uniform_strategy]
result_uniform = wbr_sim.run_simulation(num_players, uniform, strategy_choices,
                                        low_alpha, random_walk=False)
degree_array = [np.mean(result_uniform[i]) for i in range(num_players)]
# plt.plot([i for i in range(num_players)], degree_array)
# draw_result(result_uniform)

"""Log Plots"""
num_players = 1000
high_alpha = lambda x, y: 0.4
num_trials = 10
strategies = [helpers.uniform_strategy]
strategy_choices = [0 for i in range(num_players)]

result = wbr_sim.fine_tuned_simulation(num_players, num_trials, strategies, strategy_choices,
                          high_alpha, random_walk=False)

# Plot avg utility vs node number
result_mean = [helpers.avg_utility(result)[node] for node in helpers.avg_utility(result).keys()]
y_axis = [node for node in range(len(result))]
print(result_mean)
print(y_axis)
plt.plot(y_axis, result_mean)
plt.ylabel('Node Utility')
plt.xlabel('Node Position')
plt.show()
exit()

# Loglog plot
plt.plot(np.log(result_mean), np.log(y_axis), 'ro')
plt.title('Power Law, Utility as a function of Node Entry Time')
plt.ylabel('Node Entry Time')
plt.xlabel('Node Utility')
plt.show()
exit()

# degree_array = [np.mean(result[i]) for i in range(num_players)]
# print(degree_array)
# print(np.log(degree_array))
# plt.plot(np.log(degree_array), np.log([i for i in range(num_players)]), 'ro')
# plt.show()

max_degree = int(max(degree_array)) + 1
# possible_degrees = [i for i in range(1, max_degree)]
# possible_degrees = possible_degrees[::-1]


degree_counts = {}
for d in degree_array:
  if d in degree_counts:
    degree_counts[d] += 1
  else:
    degree_counts[d] = 1


print(degree_counts)
lists = sorted(degree_counts.items())

x, y = zip(*lists)

plt.loglog(x, y, 'ro')
plt.show()


