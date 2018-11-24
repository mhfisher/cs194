import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import helpers

def fine_tuned_simulation(player_range, num_trials, strategy_dict,
                          alpha_function, random_walk=False):
  """
  Run WBR simulation with more granularity for controlling input parameters.
  Returns a utility_dict where utility_dict[i] is an array with
  utility_dict[i][j] = utility of node i on game j

  :param num_trials (int): Number of trials to run for each particular set of
    inputs. Note that this will run num_trials simulations for each i in the
    player_range.

  :param strategy_functions (func array): An array of f(graph) that contains the
    various strategy functions nodes should play.

  :param strategy_subsets (float array): strategy_dict[i] should yield the
    strategy_function that node i will use in the game.

  :param alpha_function (returns 0-1 float): Should be f(node, graph) that returns
    alpha value for a particular node. The paper has uniform constant alpha for all nodes.

  :param player_range (int tuple): Simulation will be run with player_range[0]
    through player_range[1] players, therefore run player_range[1] - player-range[0] times.
    Note that if a constant num_players k is desired, tuple should be (k-1, k)

  :param random_walk: If on, deferred node accepts only with probability alpha, not 1.
  """
  graph_array = []
  for t in range(player_range[0], player_range[1]):
    for i in range(num_trials):
      graph_array.append(run_simulation(t, strategy_dict,
                                        alpha_function, random_walk))

  utility_dict = dict([(i, []) for i in range(player_range[1])])
  for graph in graph_array:
    for node in graph.keys():
      utility_dict[node].append(len(graph[node]))

  return utility_dict


def run_simulation(num_nodes, strategy_dict,
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
    strategy = strategy_dict[new_node]
    chosen_host = np.random.choice(choice_array, p=strategy(graph))

    # Add the new node to the graph
    graph.update({new_node: []})

    # Update the graph with this node's host
    graph = helpers.connect_or_defer(new_node, chosen_host, graph, alpha_function,
                                   random_walk)

  return graph



# Run the traditional game as presented in the paper, 100 players, alpha = 0.4
PA_dict = dict([(i, helpers.PA_strategy) for i in range(100)])
alpha_function = lambda x, y: 0.4
result = fine_tuned_simulation((99, 100), 1, PA_dict,
                               alpha_function, random_walk=False)

# print([(sum(result[i]) / float(len(result[i]))) if len(result[i]) != 0 \
#         else 0 for i in result.keys()])


# nx.draw(nx.Graph(graph))
# plt.draw()
# plt.show()
