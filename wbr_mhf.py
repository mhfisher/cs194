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


# num_players = 100
# strategies = [helpers.PA_strategy]
# strategy_choices = [0 for i in range(num_players)]


def draw_high_alpha_sim(num_players, strategies, strategy_choices):
  high_alpha = lambda x, y: 0.75
  result_high_alpha = wbr_sim.run_simulation(num_players, strategies, strategy_choices,
                                  high_alpha, random_walk=False)
  draw_result(result_high_alpha)


def draw_low_alpha_sim(num_players, strategies, strategy_choices):
  low_alpha = lambda x, y: 0.25
  result_low_alpha = wbr_sim.run_simulation(num_players, strategies, strategy_choices,
                                  low_alpha, random_walk=False)
  draw_result(result_low_alpha)


"""Log-Log Plots"""
num_players = 1000
alpha = lambda x, y: 0.5
num_trials = 1
strategies = [helpers.PA_strategy]
strategy_choices = [0 for i in range(num_players)]

result = wbr_sim.fine_tuned_simulation(num_players, num_trials, strategies, strategy_choices,
                                       alpha, random_walk=False)


def mean_std_dev_stats(degree_array):
  """Spit out mean and std dev for a given degree distribution"""
  result_mean = np.mean(degree_array)
  result_std_dev = np.std(degree_array)
  print('Distribution mean: {}'.format(result_mean))
  print('Distribution std dev: {}'.format(result_std_dev))
  within_std_dev = 0
  for d in degree_array:
    if d >= (result_mean - result_std_dev) and d <= (result_mean + result_std_dev):
      within_std_dev += 1

  print('Percentage within one std dev of mean: {}' \
        .format(within_std_dev / float(len(degree_array))))

def plot_cdf(utility_dict, strategy, alpha):
  """Plot CDF of node degree distribution"""
  degree_array = []
  for node in utility_dict:
    degree_array += utility_dict[node]

  degree_counts = {}
  for d in degree_array:
    degree_counts[d] = degree_counts.get(d, 0) + 1

  tuple_array = sorted(degree_counts.items())
  # tuple_array = [(i, degree_array[i]) for i in range(len(degree_array))]
  print(tuple_array)

  remaining_observations = num_players * num_trials
  cdf_array = []
  for pair in tuple_array:
    cdf_array.append(remaining_observations)
    remaining_observations = remaining_observations - pair[1]

  print(cdf_array)

  x, y = zip(*tuple_array)

  # x = x[1:]
  # cdf_array = cdf_array[1:]

  print(len(y))
  plt.loglog(x, cdf_array, 'ro')
  # plt.plot(x, cdf_array, 'ro')
  plt.xlabel('Degree Value')
  plt.ylabel('Number of Occurrences >= x')
  plt.title('CDF Plot, Strategy: {0}, Alpha: {1}'.format(strategy, alpha))
  plt.show()

strategy = strategies[0].func_name
alpha_value = alpha(0,1)
plot_cdf(result, strategy, alpha_value)
