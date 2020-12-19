import numpy as np
import matplotlib.pyplot as plt
import math

# import datetime to check time spent running script
# makes script as efficent as possible
import datetime


class Game:

    def __init__(self, height, width):

        # initialize the width and height of the grid
        self.height = height
        self.width = width

    def generateGrid(self):

        # generates a Height x Width 2d array with random elements from 0 - 9
        grid = np.random.randint(low=0, high=9, size=(self.height, self.width))
        # returns the generated grid
        return grid

    def dijkstra(self, grid, start):

        # row and col are the lengths of our 2d array (grid is the 2d array)
        row = len(grid)
        col = len(grid[0])

        # cost to each "node" FROM STARTING NODE!!!!!!!!!!. updates as we go through "nodes"
        # 2d array mirroring our grid
        # at first, the cost to get to each node is 99999999 (a lot)
        distance = np.full((row, col), 99999999)
        # the cost to our start node is 0
        distance[start] = 0

        # visited and unvisited nodes
        # 2d array mirroring our grid
        # visited node is a 1
        # unvisited node is a 0
        # at first, all nodes are unvisited, so 0
        visited = np.zeros((row, col), dtype=int)

        # set for holding nodes to check in smallestUnvisited function, so we dont check all nodes every time
        # if we had a M x N grid, we would check M x N values for the smallest unvisited one
        # with this, we improve the total time of running this script by only checking neightbours of visited nodes
        nodesToCheck = set()
        nodesToCheck.add((0, 0))

        # function to find the smallest distance node, from the unvisited nodes
        def smallestUnvisited(distance, nodesToCheck):

            # smallest distance node i coordinate
            sm_i = -1
            # smallest distance node j coordinate
            sm_j = -1

            # smallest distance node value (initial)
            sm = 99999999

            # we check every node for the smallest value
            for node in nodesToCheck:

                i, j = node

                if (distance[i][j] < sm):
                    sm = distance[i][j]
                    sm_i = i
                    sm_j = j

            # we return the coordinates of our smallest distance unvisited node
            return (sm_i, sm_j)

        # start going through all nodes in our grid and updating distances
        # while there exists nodes to go through (see function declaration above)
        while(len(nodesToCheck) != 0):

            # get the i and j of smallest distance unvisited node
            i, j = smallestUnvisited(distance, nodesToCheck)

            # for south, east, norths, west we check if there exists an unvisited node
            # we then compare the current distance for that node with
            # the distance of the current node plus the cost
            # (cost is the number of the next node)
            # if the current distance is greater, i change it to the lower value i just computed

            # south
            # if there exists a node to the south that is UNVISITED
            if i+1 < len(distance) and visited[i+1][j] == 0:
                # add node to set, to be checked later when we compute the smallest value form unvisited nodes
                nodesToCheck.add((i+1, j))
                # compute distance
                if distance[i+1][j] > grid[i+1][j] + distance[i][j]:
                    distance[i+1][j] = grid[i+1][j] + distance[i][j]

            # east
            # if there exists a node to the east that is UNVISITED
            if j+1 < len(distance[0]) and visited[i][j+1] == 0:
                # add node to set, to be checked later when we compute the smallest value form unvisited nodes
                nodesToCheck.add((i, j+1))
                # compute distance
                if distance[i][j+1] > grid[i][j+1] + distance[i][j]:
                    distance[i][j+1] = grid[i][j+1] + distance[i][j]

            # north
            if i-1 >= 0 and visited[i-1][j] == 0:
                # add node to set, to be checked later when we compute the smallest value form unvisited nodes
                nodesToCheck.add((i-1, j))
                # compute distance
                if distance[i-1][j] > grid[i-1][j] + distance[i][j]:
                    distance[i-1][j] = grid[i-1][j] + distance[i][j]

            # west
            if j-1 >= 0 and visited[i][j-1] == 0:
                # add node to set, to be checked later when we compute the smallest value form unvisited nodes
                nodesToCheck.add((i, j-1))
                # compute distance
                if distance[i][j-1] > grid[i][j-1] + distance[i][j]:
                    distance[i][j-1] = grid[i][j-1] + distance[i][j]

            # mark node as visited
            visited[i][j] = 1

            # remove current node from nodesToCheck, so we dont check it again, causing errors in the flow
            nodesToCheck.remove((i, j))

        # returning distance to bottom right cornet of 2d array
        return distance[row-1][col-1]

    def BFS(self, grid, start):

        # BFS is similar to dijskras except it only checks south and east and
        # doesnt have a way of picking which node to visit next
        # it always picks the first node in the queue

        # row and col are the lengths of our 2d array (grid is the 2d array)
        row = len(grid)
        col = len(grid[0])

        # cost to each "node" FROM STARTING NODE!!!!!!!!!!. updates as we go through "nodes"
        # 2d array mirroring our grid
        # at first, the cost to get to each node is 99999999 (a lot)
        distance = np.full((row, col), 99999999)
        # the cost to our start node is 0
        distance[start] = 0

        # data structure for keeping visited nodes, so we dont visit more than once and go into an infinite loop
        visited = np.zeros((row, col))

        # queue for checking nodes
        queue = []
        # we add first node to queue
        queue.append((0, 0))

        # while queue is not empty
        while(len(queue) != 0):

            # get coordinates of first node in queue
            i, j = queue[0]
            # remove first node from queue
            queue.pop(0)
            # mark it as visited
            visited[i][j] = 1

            # if South node exists, is not visited and not already in the queue
            if(i+1 < row and visited[i+1][j] == 0 and (i+1,j) not in queue):
                # add node to queue
                queue.append((i+1, j))
                # compute distance
                if distance[i+1][j] > grid[i+1][j] + distance[i][j]:
                    distance[i+1][j] = grid[i+1][j] + distance[i][j]

            # if East node exists
            if(j+1 < col and visited[i][j+1] == 0 and (i,j+1) not in queue):
                # add node to queue
                queue.append((i, j+1))
                # compute distance
                if distance[i][j+1] > grid[i][j+1] + distance[i][j]:
                    distance[i][j+1] = grid[i][j+1] + distance[i][j]
        
        # return distance to bottom right corner (calculated only with right and down movements)
        return distance[row-1][col-1]

    def ant_colony(self, grid, start):

        # row and col are the lengths of our 2d array (grid is the 2d array)
        row = len(grid)
        col = len(grid[0])

        # end node
        end = (row - 1, col - 1)

        # initialize pheromones (similar to weights from neural networks)
        pheromones = np.ones(shape=(row, col))
        # constant that gets divided by a distance when updating pheromones
        # used for updateing pheromones
        q_constant = 1.1
        # constant that "fades out" the pheromones
        evaporation_rate = 0.55

        # set number of generations (epochs) and ants
        ants = 256*3+32+8+16+32+128+32
        gens = 32+16+8+4+8

        # initial shortest path
        shortest_path = 99999999

        # helper functions

        # selects a node for the ant to visit
        def roulette_select(current_node, nodes_to_check):
            # nodes to check contains the neighbours of current node that meet a specific criteria (exist, not in current path)
            # n = probability
            n = np.random.uniform(0, 1)

            # sum of all activations (a)
            s = 0

            # list for nodes and probability of nodes
            prob = []
            nodes = []

            # for each node
            for node in nodes_to_check:
                # add it to nodes
                nodes.append(node)
                # create activation (a) based on distance and pheromones
                # if the pheromones are low, the activation will be low
                # if the distance is low, the activation will be high
                if(distance(current_node, node) != 0):
                    a = (1 / distance(current_node, node)) * \
                        pheromone(current_node, node)
                else:
                    a = pheromone(current_node, node)
                # add activation to sum
                s += a
                # add activation to probability list
                prob.append(a)

            prob = np.array(prob, dtype='float64')
            # divide the probability list by the sum
            # prob now contains the probability of each node to be picked
            # sum of probability list is now 1
            prob = prob / s

            # choose a node based on the probability list generated above and n
            cumulative_sum = 0
            chosen = 0
            # developed this code using the pseudocode from Wikipedia and a YouTube video
            # adapted pseudocode for my project
            for i in range(len(prob)):
                if cumulative_sum < n:
                    chosen = i
                cumulative_sum += prob[i]

            return nodes[chosen]

        # returns the pheromone levels between 2 points
        def pheromone(p1, p2):
            pher = pheromones[p2[0]][p2[1]]
            return pher

        # distance between 2 points using "The time spent on a cell is the number on this cell"
        def distance(p1, p2):
            dist = grid[p2[0]][p2[1]]
            return dist

        # update pheromones after each generation
        def update_pheromones(paths):
            # apply evaporation rate
            # the pheromones "lose" power after each generation
            new_pheromones = (1 - evaporation_rate) * pheromones

            # update each pheromone manually
            # formula found in Wikipedia
            for hist, dist in paths:
                for node in hist:
                    i = node[0]
                    j = node[1]

                    # i changed this because I cant divide by 0
                    if (dist == 0):
                        dist = 0.75

                    # update pheromones at a specific node
                    # pheromone after evaporation + a constant divided by distance traveled by the ant
                    new_node_pher = new_pheromones[i][j] + (q_constant / dist)
                    new_pheromones[i][j] = new_node_pher

            # return pheromones
            return new_pheromones

        # starting from node, return a set of new nodes for the the ant to choose from
        def update_nodes_to_check(node, path):

            i = node[0]
            j = node[1]

            new_nodes_to_check = set()

            # if node exists
            # if node not already visited
            if((i+1 < row) and ((i+1, j) not in path)):
                new_nodes_to_check.add((i+1, j))
            if((i-1 >= 0) and ((i-1, j) not in path)):
                new_nodes_to_check.add((i-1, j))
            if((j+1 < col) and ((i, j+1) not in path)):
                new_nodes_to_check.add((i, j+1))
            if((j-1 >= 0) and ((i, j-1) not in path)):
                new_nodes_to_check.add((i, j-1))

            # return the new set of nodes for roulette selection
            return new_nodes_to_check

        # if a shorter path exists, update the distance of the shortest path
        def update_shortest_path(paths):

            current_shortest = shortest_path
            # check each valid path
            # i say valid because sometimes the ant doesnt reach the end node
            #  that path is not added in the paths list
            for hist, dist in paths:
                if dist < current_shortest:
                    # update shortest distance
                    current_shortest = dist

            return current_shortest

        # for each generation
        for g in range(gens):

            # list for storing paths of that generation
            paths = []

            # for each ant
            for a in range(ants):

                # start point
                current_node = (0, 0)
                current_distance = 0

                # path of ant
                path = set()
                path.add(current_node)
                # path of ant, in the order of nodes
                path_in_order = []
                path_in_order.append(current_node)

                # nodes to check with roulette selection
                nodes_to_check = set()

                nodes_to_check.add((1, 0))
                nodes_to_check.add((0, 1))

                # if there are nodes to check and the current node is not the end node
                while (len(nodes_to_check) != 0) and (current_node != end):
                    # select next node
                    next_node = roulette_select(current_node, nodes_to_check)
                    # compute distance to next node from START of path to next node
                    current_distance += distance(current_node, next_node)
                    # create a new set of nodes to check in the next while loop
                    nodes_to_check = update_nodes_to_check(next_node, path)
                    # set current node to next node
                    current_node = next_node
                    # add node to path
                    path.add(next_node)
                    path_in_order.append(next_node)

                # the ant doesnt always reach the end node (gets lost or trapped), so we check if it found a viable path before adding to paths list
                if(end in path):
                    paths.append([path_in_order, current_distance])

            # update pheromones and shortest path for next generation
            pheromones = update_pheromones(paths)
            shortest_path = update_shortest_path(paths)

        # returns the shortest path to end node
        return shortest_path

# testing starts here

grid2 = [[1, 9, 9, 9],
         [1, 9, 9, 9],
         [1, 9, 9, 9],
         [1, 1, 1, 1]]

grid6 = [[1, 9, 9],
         [1, 9, 9],
         [1, 1, 1]]

grid3 = [[1, 4, 1],
         [1, 2, 1]]

grid4 = [[0, 9, 9, 9, 9],
         [0, 9, 0, 0, 0],
         [0, 9, 0, 9, 0],
         [0, 9, 0, 9, 0],
         [0, 0, 0, 9, 0]]

grid5 = [[0, 9, 0, 0, 0, 0],
         [0, 9, 0, 9, 9, 0],
         [0, 9, 0, 0, 9, 0],
         [0, 9, 9, 0, 9, 0],
         [0, 0, 0, 0, 9, 0]]

grid7 = [[0, 6, 4, 5, 1, 4, 3, 5, 6, 8, 7],
         [1, 3, 3, 9, 1, 4, 3, 5, 6, 2, 1],
         [4, 1, 9, 1, 1, 4, 3, 5, 6, 5, 3],
         [9, 6, 1, 2, 1, 4, 3, 5, 6, 2, 1],
         [1, 3, 5, 4, 1, 4, 3, 5, 6, 8, 4],
         [8, 7, 2, 9, 1, 4, 3, 5, 6, 7, 5],
         [1, 6, 3, 5, 1, 4, 3, 5, 6, 2, 2],
         [8, 7, 2, 9, 1, 4, 3, 5, 6, 7, 5],
         [1, 6, 3, 5, 1, 4, 3, 5, 6, 2, 2],
         [8, 7, 2, 9, 1, 4, 3, 5, 6, 7, 5],
         [1, 6, 3, 5, 1, 4, 3, 5, 6, 2, 2]]

grid8 = [[1, 9, 9, 9, 9, 9],
         [1, 1, 9, 1, 1, 1],
         [9, 1, 9, 1, 9, 1],
         [9, 1, 9, 1, 9, 1],
         [9, 1, 9, 1, 9, 1],
         [9, 1, 9, 1, 9, 1],
         [9, 1, 1, 1, 9, 1]]

grid9 = [[0, 6, 4, 5],
         [1, 3, 3, 9],
         [4, 9, 2, 1],
         [9, 6, 1, 2],
         [2, 3, 4, 5]]

game = Game(14, 14)
grid_genrated = game.generateGrid()

grid = grid_genrated

print('\n')
# compute distance with Dijkstra
begin_time = datetime.datetime.now()
distance = game.dijkstra(grid, (0, 0))
print("time     - Dijkstra ", datetime.datetime.now() - begin_time)
print("distance - Dijkstra ", distance)

print('\n')
print("ACO started")
# compute distance with ant colony
begin_time = datetime.datetime.now()
distance3 = game.ant_colony(grid, (0, 0))
print("time     - ant_colony ", datetime.datetime.now() - begin_time)
print("distance - ant_colony ", distance3)

print('\n')
# compute distance with BFS
begin_time = datetime.datetime.now()
distance2 = game.BFS(grid, (0, 0))
print("time     - BFS ", datetime.datetime.now() - begin_time)
print("distance - BFS ", distance2)