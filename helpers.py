"""Helper functions"""
import numpy as np
import random

def connect_or_defer(requester, host, graph, alpha_function, random_walk=False):
  """
  Assigns requester to host with probability alpha.
  Assigns requester to one of host's neighbors with probability (1 - alpha).
  Choice of host's neighbor is made uniformly at random.

  If random_walk flag is set, the host's neighbor also accepts with probability
  alpha, and recursively defers with (1 - alpha).

  Returns the new state of the graph.
  """
  rand = random.random()
  alpha = alpha_function(host, graph)
  if rand <= alpha:
    graph.get(requester).append(host)
    graph.get(host).append(requester)
  else:
    # Choose a random neighbor to defer to
    defer_node = random.randint(0, len(graph[host]))
    if random_walk:
      # We continue the random walk with probability (1 - alpha)
      return connect_or_defer(requester, defer_node, graph, alpha_function, random_walk)
    if not random_walk:
      # Host must accept
      graph.get(requester).append(defer_node)
      graph.get(defer_node).append(requester)

  return graph

def comparative_utility(graph1, graph2, subset):
  """
  Returns the difference in utility (degree) of the subset nodes in graph 1 versus graph 2.
  Subset should be a list of nodes (integers) of interest that will be compared
  across the two graphs.
  """
  utlity_dict = dict()
  for node in subset:
    utility_dict[node] = graph1.degree(node) - graph2.degree(node)
  return utility_dict

def avg_utility(utility_dict):
  """
  Return avg_utility_dict
  """
  return dict([(key, np.mean(utility_dict[key])) for key in utility_dict])

def total_utility(utility_dict):
  """
  Return total utility dict
  """
  return dict([(key, sum(utility_dict[key])) for key in utility_dict])

# Strategy functions

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

