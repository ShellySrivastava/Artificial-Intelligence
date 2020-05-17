import networkx as nx

from UCS import load_graph_from_file
from UCS import uniform_cost_search
from UCS import construct_path


def evaluate_cost(uk_cities):
    # velocity for optimal path (kmph)
    velocity = 316.22
    # air pollution rate (per hour)
    air_pollution = 0.00001 * (velocity ** 2)
    # get all roads
    all_roads = uk_cities.edges()
    for i, val in enumerate(all_roads):
        # get the distance of the road
        distance = uk_cities[val[0]][val[1]]['weight']
        time = distance/velocity
        # calculate the cost
        cost = time + (air_pollution*time)
        # update the weight of the graph for UCS
        uk_cities[val[0]][val[1]]['weight'] = cost
    return uk_cities


# load json file
uk_cities = load_graph_from_file("UK_cities.json")
# get the updated graph with new costs
uk_environment_cost = evaluate_cost(uk_cities)
# get the solution path
solution = uniform_cost_search(uk_environment_cost, 'london', 'aberdeen')
print("Total cost: ", solution.path_cost)
print("Path:", construct_path(solution))
