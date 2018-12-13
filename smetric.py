import helpers
import wbr_sim

def s_max(degree_sequence):
    """
    Returns a graph with the given degree sequence that maximizes the s_metric

    :param degree_sequence (int list): desired degree sequence of the graph sorted in descending order.
    """
    # Step 0: Initialization
    A = {0: []}
    B = [i for i in range(1, len(degree_sequence))]
    O = []
    for i in range(len(degree_sequence)):
        for j in range(i+1, len(degree_sequence)):
            O.append((i,j))
    O = sorted(O, key=lambda x: degree_sequence[x[0]]*degree_sequence[x[1]], reverse=True)
    w_hat = [i for i in degree_sequence]
    wA = degree_sequence[0]
    dB = sum(degree_sequence)-wA
    # Step 1: Link Selection
    while O:
        # choose links with largest weight d_i*d_j
        admissible = [O[0]]
        for i in range(1, len(O)):
            if O[0][0]*O[0][1] == O[i][0]*O[i][1]:
                admissible.append(O[i])
        # filter out inadmissible links
        for link in admissible.copy():
            if w_hat[link[0]] == 0 or w_hat[link[1]] == 0:
                O.remove(link)
                admissible.remove(link)
        # return to step 1 if there are no admissible links
        if not admissible:
            continue
        # sort the links by largest w_i, breaking ties with the value of w_j
        admissible = sorted(admissible, key=lambda x: 10*w_hat[x[0]] + w_hat[x[1]], reverse=True)
        link_to_add = admissible[0]
        i, j = link_to_add
        # Step 2: Link Addition
        if i in A and j in B:
            # Type 1: i in A and j in B
            A[i].append(j)
            A[j] = [i]
            B.remove(j)
            w_hat[i] -= 1
            w_hat[j] -= 1
            wA += (degree_sequence[j]-2)
            dB -= degree_sequence[j]
            O.remove(link_to_add)
        else:
            # Type 2: i in A and j in A
            # Check Tree Condition
            if dB == 2*len(B) - wA:
                O.remove(link_to_add)
            # Check Disconnected Cluster Condition
            elif wA == 2:
                O.remove(link_to_add)
            else:
                A[i].append(j)
                A[j].append(i)
                w_hat[i] -= 1
                w_hat[j] -= 1
                wA -= 2
                O.remove(link_to_add)
    return A

def s_metric(graph):
    """
    Returns the "scale-free metric" from Li et al. (2005)

    :param graph (int dict): a dictionary representing an a graph as an adjacency list
    """
    total = 0
    for u in graph:
        for v in graph[u]:
            total += len(graph[u])*len(graph[v])
    return total/2

def deg_seq(graph):
    """
    Returns the degree sequence of the graph in descending order

    :param graph (int dict): a dictionary representing an a graph as an adjacency list
    """
    seq = []
    for u in graph:
        seq.append(len(graph[u]))
    return sorted(seq, reverse=True)

def scale_free(graph):
    """
    Returns the a number between 0 and 1 representing how scale free a graph is
    Values closer to 0 are "scale-rich" while values closer to 1 are "scale-free"

    :param graph (int dict): a dictionary representing an a graph as an adjacency list
    """
    return s_metric(graph)/s_metric(s_max(deg_seq(graph)))
