import networkx as nx

from UCS import load_graph_from_file
from UCS import uniform_cost_search
from UCS import construct_path


# load json file
uk_cities = load_graph_from_file("UK_cities.json")
# get the solution
solution = uniform_cost_search(uk_cities, 'london', 'aberdeen')
print("Total cost: ", solution.path_cost)
print("Path:", construct_path(solution))
