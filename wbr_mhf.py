import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.stats

import helpers
import wbr_sim


def draw_result(result):
    """ Draw simulation result"""
    nx.draw_kamada_kawai(nx.Graph(result), node_size=100)
    plt.show()


def draw_high_alpha_sim(num_players, strategies, strategy_choices):
    high_alpha = 1
    result_high_alpha = wbr_sim.run_simulation(
        num_players,
        strategies,
        strategy_choices,
        high_alpha,
        random_walk=False)
    draw_result(result_high_alpha)


def draw_low_alpha_sim(num_players, strategies, strategy_choices):
    low_alpha = 0.25
    result_low_alpha = wbr_sim.run_simulation(
        num_players,
        strategies,
        strategy_choices,
        low_alpha,
        random_walk=False)
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
    degree_array = []
    for node in utility_dict:
        degree_array += utility_dict[node]

    degree_counts = {}
    for d in degree_array:
        degree_counts[d] = degree_counts.get(d, 0) + 1

    tuple_array = sorted(degree_counts.items())
    print(tuple_array)

    total_observations = num_players * num_trials
    remaining_observations = num_players * num_trials
    cdf_array = []
    for pair in tuple_array:
        cdf_array.append(remaining_observations)
        remaining_observations = remaining_observations - pair[1]

    x, y = zip(*tuple_array)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    plt.loglog(x, cdf_array, color)
    plt.loglog(x, [intercept + (slope*x_i) for x_i in x], 'b')
    print(slope)
    print(intercept)
    print(r_value)
    print(p_value)
    # plt.loglog(degree_array, [degree_array[i] for i in degree_array], color)
    plt.xlabel('Degree Value')
    plt.ylabel('Number of Occurrences >= x')
    plt.title('CDF Plot, Strategy: {0}, Alpha: {1}'.format(strategy, alpha))
    plt.show()


num_players = 1000
alpha = 1
num_trials = 1
strategies = [helpers.PA_strategy]
strategy_choices = [0 for i in range(num_players)]

result = wbr_sim.fine_tuned_simulation(
    num_players,
    num_trials,
    strategies,
    strategy_choices,
    alpha,
    random_walk=False)
print(result)
num_over_threshold = 0
# for node in result:
#     neighbors = result[node]
#     one_over_degree = [(1/len(result[neighbor])) for neighbor in neighbors]
#     ratio = sum(one_over_degree) / len(neighbors)
#     if ratio >= 0.5*(1/(1-alpha)):
#         print(node)
#         print(ratio)
#         num_over_threshold += 1
# print(num_over_threshold)

strategy = strategies[0].__name__
plot_cdf(result, strategy, alpha)
plt.show()
exit()


def nodewise_diff(alpha, num_players, num_trials):
    """
  Return the MAGNITUDE of the node-by-node utility difference for
  playing alpha vs PA for a given alpha.
  """
    PA_dict = wbr_sim.fine_tuned_simulation(
        num_players,
        num_trials, [helpers.PA_strategy], [0 for i in range(num_players)],
        alpha,
        random_walk=False)
    uniform_dict = wbr_sim.fine_tuned_simulation(
        num_players,
        num_trials, [helpers.uniform_strategy],
        [0 for i in range(num_players)],
        alpha,
        random_walk=False)

    PA_distribution, uniform_distribution = [], []
    for node in PA_dict.keys():
        PA_distribution += PA_dict[node]
        uniform_distribution += uniform_dict[node]

    max_norm = sum(PA_distribution)

    return ((np.linalg.norm(PA_distribution) / max_norm),
            (np.linalg.norm(uniform_distribution) / max_norm))

    diff_array = []
    for n in range(num_players):
        diff_array += [(PA_dict[n][i] - uniform_dict[n][i])
                       for i in range(len(PA_dict[n]))]

    better_off = [value if value > 0 else 0 for value in diff_array]
    worse_off = [value if value < 0 else 0 for value in diff_array]

    utility_diff = np.linalg.norm(better_off) - np.linalg.norm(worse_off)
    return utility_diff
