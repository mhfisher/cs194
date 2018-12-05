import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import helpers
import wbr_sim


def draw_result(result):
  """ Draw simulation result"""
  nx.draw_kamada_kawai(nx.Graph(result), node_size=100)
  plt.show()


def draw_high_alpha_sim(num_players, strategies, strategy_choices):
  high_alpha = lambda x, y: 1
  result_high_alpha = wbr_sim.run_simulation(
      num_players, strategies, strategy_choices, high_alpha, random_walk=False)
  draw_result(result_high_alpha)


def draw_low_alpha_sim(num_players, strategies, strategy_choices):
  low_alpha = lambda x, y: 0.25
  result_low_alpha = wbr_sim.run_simulation(
      num_players, strategies, strategy_choices, low_alpha, random_walk=False)
  draw_result(result_low_alpha)


"""Log-Log Plots"""


def mean_std_dev_stats(degree_array):
  """Spit out mean and std dev for a given degree distribution"""
  result_mean = np.mean(degree_array)
  result_std_dev = np.std(degree_array)
  print('Distribution mean: {}'.format(result_mean))
  print('Distribution std dev: {}'.format(result_std_dev))
  within_std_dev = 0
  for d in degree_array:
    if d >= (result_mean - result_std_dev) and d <= (
        result_mean + result_std_dev):
      within_std_dev += 1

  print('Percentage within one std dev of mean: {}' \
        .format(within_std_dev / float(len(degree_array))))


def plot_cdf(utility_dict, strategy, alpha, color='ro'):
  """Plot CDF of node degree distribution"""
  print(alpha)
  degree_array = []
  for node in utility_dict:
    # degree_array.append(np.median(utility_dict[node]))
    degree_array += utility_dict[node]

  degree_counts = {}
  for d in degree_array:
    degree_counts[d] = degree_counts.get(d, 0) + 1

  tuple_array = sorted(degree_counts.items())
  # tuple_array = [(i, degree_array[i]) for i in range(len(degree_array))]
  print(tuple_array)

  total_observations = num_players * num_trials
  remaining_observations = num_players * num_trials
  cdf_array = []
  for pair in tuple_array:
    cdf_array.append(remaining_observations / total_observations)
    remaining_observations = remaining_observations - pair[1]

  print(cdf_array)

  x, y = zip(*tuple_array)

  # x = x[1:]
  # cdf_array = cdf_array[1:]

  print(len(y))
  plt.loglog(x, cdf_array, color)
  # plt.plot(np.log(x, 10), np.log(cdf_array, 10), 'ro')
  plt.xlabel('Degree Value')
  plt.ylabel('Number of Occurrences >= x')
  plt.xlim(right=1000)
  plt.title('CDF Plot, Strategy: {0}, Alpha: {1}'.format(strategy, alpha))
  # plt.show()


num_players = 1000
alpha = lambda x, y: 0.25
num_trials = 1
strategies = [helpers.PA_strategy]
strategy_choices = [0 for i in range(num_players)]

result = wbr_sim.fine_tuned_simulation(
    num_players,
    num_trials,
    strategies,
    strategy_choices,
    alpha,
    # Results are very interesting if you turn on random walk
    random_walk=False)

result2 = wbr_sim.fine_tuned_simulation(
    num_players,
    num_trials,
    [helpers.uniform_strategy],
    strategy_choices,
    alpha,
    random_walk=False)

strategy = strategies[0].__name__
alpha_value = alpha(0,1)
plot_cdf(result, strategy, alpha_value)
# plot_cdf(result2, strategy, alpha_value, color='bo')
plt.show()
exit()
# plt.show()
# exit()

def nodewise_diff(alpha, num_players, num_trials):
  """
  Return the MAGNITUDE of the node-by-node utility difference for
  playing alpha vs PA for a given alpha.
  """
  alpha_function = lambda x, y: alpha
  PA_dict = wbr_sim.fine_tuned_simulation(
      num_players,
      num_trials, [helpers.PA_strategy], [0 for i in range(num_players)],
      alpha_function,
      random_walk=False)
  uniform_dict = wbr_sim.fine_tuned_simulation(
      num_players,
      num_trials, [helpers.uniform_strategy], [0 for i in range(num_players)],
      alpha_function,
      random_walk=False)

  PA_distribution, uniform_distribution = [], []
  for node in PA_dict.keys():
    PA_distribution += PA_dict[node]
    uniform_distribution += uniform_dict[node]

  max_norm = sum(PA_distribution)

  return ((np.linalg.norm(PA_distribution) / max_norm), (np.linalg.norm(uniform_distribution) / max_norm))

  diff_array = []
  for n in range(num_players):
    diff_array += [(PA_dict[n][i] - uniform_dict[n][i])
                   for i in range(len(PA_dict[n]))]

  better_off = [value if value > 0 else 0 for value in diff_array]
  worse_off = [value if value < 0 else 0 for value in diff_array]

  utility_diff = np.linalg.norm(better_off) - np.linalg.norm(worse_off)
  return utility_diff


def s_metric(graph):
  """
  Build on networkx's s_metric implementation to yield a modified version
  of Li et. al's s-metric.
  DOES NOT NORMALIZE
  """
  # degree_distribution = [node[1] for node in graph.degree()]
  s_value = lambda G: float(sum([G.degree(u) * G.degree(v) for (u, v) in G.edges()]))

  return s_value(nx.Graph(graph))

  # random_g_array = [nx.expected_degree_graph(degree_distribution, selfloops=False) for i in range(1)]
  # avg_s_value = np.mean([s_value(g) for g in random_g_array])
  # print(avg_s_value)
  # curr_s_value = s_value(graph)
  # print(curr_s_value)

  # return (curr_s_value / avg_s_value)

def s_metric_compare(alpha, num_players, num_trials, ideal):
  alpha_function = lambda x, y: alpha
  PA_dict = wbr_sim.run_simulation(
      num_players,
      [helpers.PA_strategy], [0 for i in range(num_players)],
      alpha_function,
      random_walk=False)
  degree_seq = [len(PA_dict[node]) for node in PA_dict.keys()]
  hope = nx.generators.degree_seq.li_smax_graph(degree_seq)
  nx.draw(hope)
  uniform_dict = wbr_sim.run_simulation(
      num_players,
      [helpers.uniform_strategy], [0 for i in range(num_players)],
      alpha_function,
      random_walk=False)

  PA_s_metric = s_metric(nx.Graph(PA_dict)) / ideal
  uniform_s_metric = s_metric(nx.Graph(uniform_dict)) / ideal
  return (PA_s_metric, uniform_s_metric)

alpha_range = [(i / 10) for i in range(11)]
utility_diffs = []
num_players = 1000
num_trials = 5
strategy_choices = [0 for i in range(num_players)]

baseline = wbr_sim.run_simulation(
                                        num_players,
                                        [helpers.PA_strategy], [0 for i in range(num_players)],
                                        lambda x, y: 1,
                                        random_walk=False)
ideal_s_metric = s_metric(baseline)

for alpha in [0]:
  # print(s_metric_compare(alpha, 1000, num_trials, ideal_s_metric))
  print(nodewise_diff(alpha, 1000, num_trials))

