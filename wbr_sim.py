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
  # PA_array = [helpers.PA_strategy]
  unif_array = [helpers.uniform_strategy]
  strategy_array = [0 for i in range(num_players)]
  result = fine_tuned_simulation(num_players, 100, unif_array, strategy_array,
                                 alpha, random_walk=True)

  avg_utility_array = [helpers.avg_utility(result)[i] for i in range(len(result))]

  return avg_utility_array


def optimal_PA(num_players, num_trials, alpha, odd_node):
  """
  Have one node not play PA. Assess the change in utility if that node plays
  PA instead.
  """
  # strategy_funcs = [helpers.PA_strategy, helpers.pick_leaf]
  strategy_funcs = [helpers.PA_strategy]

  strategy_choices = [0 for i in range(num_players)]
  # strategy_choices[odd_node] = 1

  result_graph = fine_tuned_simulation(num_players, num_trials, strategy_funcs, strategy_choices,
                        alpha, random_walk=False)

  non_PA_avg_utility = np.mean(result_graph[odd_node])

  strategy_choices[odd_node] = 0
  result_graph = fine_tuned_simulation(num_players, num_trials, strategy_funcs, strategy_choices,
                        alpha, random_walk=False)

  PA_avg_utiity = np.mean(result_graph[odd_node])

  print('Number of nodes: {}'.format(num_players))
  print('Number of trials: {}'.format(num_trials))
  print('Uniform Random Choice avg utility: {}'.format(non_PA_avg_utility))
  print('PA Strategy avg utility {}'.format(PA_avg_utiity))


alpha = lambda x, y: 0.3
# optimal_PA(10000, 100, alpha, 10)
