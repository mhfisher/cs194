import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import helpers

def fine_tuned_simulation(num_players, num_trials, strategy_functions, strategy_choices,
                          alpha_function, random_walk=False):
  """
  Run WBR simulation with more granularity for controlling input parameters.
  Returns a utility_dict where utility_dict[i] is an array with
  utility_dict[i][j] = utility of node i on game j

  :param num_trials (int): Number of trials to run for each particular set of
    inputs. Note that this will run num_trials simulations for each i in the
    player_range.

  :param strategy_functions (func array): An array of f(graph) that contains the
    various strategy functions nodes can play.

  :param strategy_choices (int array): strategy_functions[strategy_choices[i]]
    yields the strategy that node i will play

  :param alpha_function (returns 0-1 float): Should be f(node, graph) that returns
    alpha value for a particular node. The paper has uniform constant alpha for all nodes.

  :param player_range (int tuple): Simulation will be run with player_range[0]
    through player_range[1] players, therefore run player_range[1] - player-range[0] times.
    Note that if a constant num_players k is desired, tuple should be (k-1, k)

  :param random_walk: If on, deferred node accepts only with probability alpha, not 1.
  """
  graph_array = []
  for i in range(num_trials):
    graph_array.append(run_simulation(num_players, strategy_functions, strategy_choices,
                                        alpha_function, random_walk))

  utility_dict = dict([(i, []) for i in range(num_players)])
  for graph in graph_array:
    for node in graph.keys():
      utility_dict[node].append(len(graph[node]))

  # Last node has zero utility and makes calculations confusing
  utility_dict.pop(num_players - 1)

  return utility_dict


def run_simulation(num_nodes, strategy_functions, strategy_choices,
                   alpha_function, random_walk=False):
  """
  Method to run WBR simulation.
  Now a helper function for fine_tuned_simulation.
  """
  # Initialize graph with first 2 turns.
  graph = {0: [1], 1: [0]}

  # Turn 3 (arbitrary choice)
  graph.update({2: []})
  graph = helpers.connect_or_defer(2, 0, graph, alpha_function) if random.random() < 0.5 else \
          helpers.connect_or_defer(2, 1, graph, alpha_function)

  # Turn 4 onwards
  for new_node in range(3, num_nodes):
    # We choose which host the new node will request
    choice_array = [i for i in range(new_node)]
    strategy = strategy_functions[strategy_choices[new_node]]
    chosen_host = np.random.choice(choice_array, p=strategy(graph))

    # Add the new node to the graph
    graph.update({new_node: []})

    # Update the graph with this node's host
    graph = helpers.connect_or_defer(new_node, chosen_host, graph, alpha_function,
                                   random_walk)

  return graph


def paper_simulation(num_players, alpha):
  """
  Run the traditional game as presented in the paper.
  Returns average utility array and returns avg_utility_array
  """
  PA_dict = [helpers.PA_strategy]
  strategy_array = [0 for i in range(100)]
  result = fine_tuned_simulation(100, 10, PA_dict, strategy_array,
                                 alpha_function, random_walk=False)

  avg_utility_array = [helpers.avg_utility(result)[i] for i in range(len(result))]

  return avg_utility_array


def PA_vs_uniform(num_players, num_trials, alpha, PA_fraction):
  """
  Run the simulation with PA_fraction * num_players playing PA,
  (1 - PA_fraction) * num_players choose a host uniformly.
  """
  uniform_strategy = lambda graph: [(1.0 / len(graph)) for i in range(len(graph))]
  strategy_funcs = [helpers.PA_strategy, uniform_strategy]

  utility_diff_over_time = [0 for i in range(num_players)]
  strategy_pool = np.random.choice([0, 1], p=[float(PA_fraction), 1.0 - PA_fraction],
                    size=num_players*num_trials).reshape(num_trials, num_players)

  for i in range(num_trials):
    strategy_choices = strategy_pool[i]


    # Run this strategy array for 10 trials.
    utility_dict = helpers.total_utility(fine_tuned_simulation(num_players, 100, strategy_funcs,
                                         strategy_choices, alpha, random_walk=False))

    # Now we reverse strategies to calculate difference in utilities.
    strategy_choices = [1 if i == 0 else 0 for i in range(len(strategy_choices))]

    flipped_utility_dict = helpers.total_utility(fine_tuned_simulation(num_players, 100,
                                         strategy_funcs, strategy_choices,
                                         alpha, random_walk=False))

    utility_diff = []
    for i in range(len(utility_dict)):
      if utility_dict[i] == helpers.PA_strategy:
        utility_diff.append(utility_dict[i] - flipped_utility_dict[i])
      else:
        utility_diff.append(flipped_utility_dict[i] - utility_dict[i])

    utility_diff_over_time = [utility_diff_over_time[i] + utility_diff[i] \
                                for i in range(len(utility_diff))]

  # Plot utility difference by node
  plt.plot(utility_diff_over_time)
  print(utility_diff_over_time)
  print(sum(utility_diff_over_time))
  plt.show()

alpha = lambda x, y: 0.5
PA_vs_uniform(100, 10, alpha, 0.5)
# Print average difference in utility
# print(utility_diff_array)
# print(sum(utility_diff_array))

# nx.draw(nx.Graph(graph))
# plt.draw()
# plt.show()
