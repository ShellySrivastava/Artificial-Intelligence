import json
import networkx as nx
from queue import PriorityQueue


class Node:
    """
    a class that defines the node for the nx graph.
    it consists pf city label, path cost to reach the city from the root and its parent node
    """
    def __init__(self, label, path_cost, parent):
        self.label = label
        self.path_cost = path_cost
        self.parent = parent

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __repr__(self):
        path = construct_path(self)
        return ('(%s, %s, %s)'
                % (repr(self.label), repr(self.path_cost), repr(path)))


def load_graph_from_file(filename):
    """
    this method loads the json file to the networkx object
    :param filename: json filename (string)
    :return: networkx object
    """
    with open(filename) as UK_cities:
        dict_uk_cities = json.load(UK_cities)
    return nx.Graph(dict_uk_cities)


def construct_path(node):
    """
    this method constructs the path as a list from the root to the node
    :param node: a Node object
    :return: list
    """
    path_from_root = [node.label]
    while node.parent:
        node = node.parent
        path_from_root = [node.label] + path_from_root
    return path_from_root


def remove_node_with_higher_cost(new_node, frontier):
    """
    this method removes the node from the priority queue if the new_node has a the same label but a lesser cost
    :param new_node: node
    :param frontier: priority queue
    :return: priority queue
    """
    removed = False
    frontier_list = frontier.queue
    for item in frontier_list:
        if item.label == new_node.label and item.path_cost > new_node.path_cost:
            removed_item = item
            frontier_list.remove(item)
            removed = True
            break

    if removed:
        print("frontier = frontier - {} + {} ".format(removed_item, new_node))
        new_queue = PriorityQueue()
        frontier_list.append(new_node)
        for item in frontier_list:
            new_queue.put(item)
        return new_queue
    else:
        return frontier


def in_frontier(new_node, frontier):
    """
    this method checks if the new_node.label is already present in the frontier
    :param new_node: node
    :param frontier: priority queue
    :return: boolean
    """
    frontier_list = frontier.queue
    for item in frontier_list:
        if item.label == new_node.label:
            return True
    return False


def uniform_cost_search(nxobject, initial, goal):
    """
    this method performs the uniform cost search
    :param nxobject: the weighted networkx graph
    :param initial: the initial state or root
    :param goal: the goal state or the destination
    :return: a node with the optimal path
    """
    node = Node(initial, 0, None)
    # frontier is a priority queue
    frontier = PriorityQueue()
    # add the initial state to the priority queue
    frontier.put(node)
    # explored is a set
    explored = set()
    print("frontier = ", frontier.queue)
    print("explored = ", explored)
    while not frontier.empty():
        print("\n")
        # pop the first element from the priority queue (lowest cost node)
        node = frontier.get()
        print("frontier = frontier - ", node)
        # check if the node is the goal state then return node
        if node.label == goal:
            return node
        # else add the node to the explored set
        explored.add(node.label)
        print("explored = explored + ", node.label)
        # get all the neighbours of the node
        neighbours = nxobject.neighbors(node.label)
        for child_label in neighbours:
            step_cost = nxobject.edges[(node.label, child_label)]['weight']
            child = Node(child_label, node.path_cost + step_cost, node)
            # check if the child node is already explored or not
            if child_label not in explored and not in_frontier(child, frontier):
                # add the child to the frontier
                frontier.put(child)
                print("frontier = frontier + ", child)
            # if the node already exists in the frontier with a higher cost, then replace it
            elif in_frontier(child, frontier):
                frontier = remove_node_with_higher_cost(child, frontier)
