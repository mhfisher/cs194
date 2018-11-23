import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def run_simulation(num_nodes, strategy_function, alpha, random_walk=False):
  """
  Method to run WBR simulation, following the rules outlined in the paper.
  Assumes that each player uses the same strategy function.
  Also assumes unchanging alpha.
  TODO: Make alpha a function of the graph

  :param num_nodes (int): Number of players (equivalently number of turns)
    Should be >= 4
  :param strategy_profile (method or lambda, int-output):
    strategy_function(graph) should yield an int_array of attachment probabilities
    for a new node arriving onto the graph. Should have sum(int_array) == 1
  :param alpha: Societal wealth constant alpha.
  """
  # Initialize graph with first 2 turns.
  graph = {0: [1], 1: [0]}

  # Turn 3 (arbitrary choice)
  graph.update({2: []})
  graph = connect_or_defer(2, 0, graph, alpha) if random.random() < 0.5 else \
            connect_or_defer(2, 1, graph, alpha)

  # Turn 4 onwards
  for new_node in range(3, num_nodes):
    # We choose which host the new node will request
    choice_array = [i for i in range(new_node)]
    chosen_host = np.random.choice(choice_array, p=strategy_function(graph))

    # Add the new node to the graph
    graph.update({new_node: []})

    # Update the graph with this node's host
    graph = connect_or_defer(new_node, chosen_host, graph, alpha, random_walk)

  return graph


def connect_or_defer(requester, host, graph, alpha, random_walk=False):
  """
  Assigns requester to host with probability alpha.
  Assigns requester to one of host's neighbors with probability (1 - alpha).
  Choice of host's neighbor is made uniformly at random.

  If random_walk flag is set, the host's neighbor also accepts with probability
  alpha, and recursively defers with (1 - alpha).

  Returns the new state of the graph.
  """
  rand = random.random()
  if rand <= alpha:
    graph.get(requester).append(host)
    graph.get(host).append(requester)
  else:
    # Choose a random neighbor to defer to
    defer_node = random.randint(0, len(graph[host]))
    if random_walk:
      # We continue the random walk with probability (1 - alpha)
      return connect_or_defer(requester, defer_node, graph, alpha, random_walk)
    if not random_walk:
      # Host must accept
      graph.get(requester).append(defer_node)
      graph.get(defer_node).append(requester)

  return graph

def PA_strategy(graph):
  """
  Yields a probability distribution over nodes following the equation:
    P[choosing node] = degree(node) / 2*(num_nodes - 1)
  where 2*(num_nodes - 1) = total_graph_degree
  """
  num_nodes = len(graph)
  probability_array = [(len(graph[node]) / (2.0 * (num_nodes - 1)))
                       for node in range(num_nodes)]

  return probability_array


graph = run_simulation(100, PA_strategy, 0.05, random_walk=True)
print(graph)

nx.draw(nx.Graph(graph))
plt.draw()
plt.show()
