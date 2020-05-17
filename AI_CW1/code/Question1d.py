import math
import random
import networkx as nx

from UCS import construct_path
from UCS import load_graph_from_file
from UCS import uniform_cost_search


def evaluate_cost(uk_cities):
    """
    this method evaluates the cost for the graph
    :param uk_cities: networkx graph
    :return: networkx graph
    """
    # get all the roads from the graph
    all_roads = uk_cities.edges()
    for i, val in enumerate(all_roads):
        # get the distance and the speed limit for the road
        distance = vlim = uk_cities[val[0]][val[1]]['weight']
        fine = 0
        # select a velocity for the given road
        v = vlim if vlim <= 300 else 300
        time = distance/v
        # check if the selected velocity is greater than speed limit
        if v > vlim:
            # check whether the person is fined or not
            if random.uniform(0, 1) <= (1 - math.exp(-v+vlim)):
                fine = 1000
        # calculate the total cost
        cost = fine + (100*time)
        # update the graph weight for UCS
        uk_cities[val[0]][val[1]]['weight'] = cost
    return uk_cities


# load json file to networkx graph
uk_cities = load_graph_from_file("UK_cities.json")
# get the graph with updated cost
uk_car_cost = evaluate_cost(uk_cities)
# get the solution path
solution = uniform_cost_search(uk_car_cost, 'london', 'aberdeen')
print("Solution: ", solution)
print(construct_path(solution))
