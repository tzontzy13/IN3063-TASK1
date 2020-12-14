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

            # for south and east, we check if there exists an unvisited node
            # we then compare the current distance for that node with
            # the distance of the current node plus the cost
            # (cost is the number of the node we are currently in)
            # if the current distance is greater, we change it to the lower value

            # south
            # if there exists a node to the south that is UNVISITED
            if i+1 < len(distance) and visited[i+1][j] == 0:
                # add node to set, to be checked later when we compute the smallest value form unvisited nodes
                nodesToCheck.add((i+1, j))
                # compute distance
                # abs(grid[i][j] - )
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

        #print("grid: \n", grid)
        #print("visited: \n", visited)
        print("distance: \n", distance)

        # returning distance to bottom right cornet of 2d array
        return distance[row-1][col-1]

    def BFS(self, grid, start):

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
            visited[i][j] = 1

            # # if North node exists
            # if(i-1 >= 0 and visited[i-1][j] == 0 and (i-1,j) not in queue):
            #     # print("ce plm north")
            #     # add node to queue
            #     queue.append((i-1, j))
            #     # compute distance
            #     if distance[i-1][j] > grid[i-1][j] + distance[i][j]:
            #         distance[i-1][j] = grid[i-1][j] + distance[i][j]

            # # if West node exists
            # if(j-1 >= 0 and visited[i][j-1] == 0 and (i,j-1) not in queue):
            #     # print("ce plm west")
            #     # add node to queue
            #     queue.append((i, j-1))
            #     # compute distance
            #     if distance[i][j-1] > grid[i][j-1] + distance[i][j]:
            #         distance[i][j-1] = grid[i][j-1] + distance[i][j]

            # if South node exists
            if(i+1 < row and visited[i+1][j] == 0 and (i+1,j) not in queue):
                # print("ce plm south")
                # add node to queue
                queue.append((i+1, j))
                # compute distance
                if distance[i+1][j] > grid[i+1][j] + distance[i][j]:
                    distance[i+1][j] = grid[i+1][j] + distance[i][j]

            # if East node exists
            if(j+1 < col and visited[i][j+1] == 0 and (i,j+1) not in queue):
                # print("ce plm east")
                # add node to queue
                queue.append((i, j+1))
                # compute distance
                if distance[i][j+1] > grid[i][j+1] + distance[i][j]:
                    distance[i][j+1] = grid[i][j+1] + distance[i][j]

            # print("q:  ",queue)
        print(distance)
        
        # return distance to bottom right corner
        return distance[row-1][col-1]

    def ant_colony(self, grid, start):

        row = len(grid)
        col = len(grid[0])

        end = (row - 1, col - 1)

        pheromones = np.ones(shape=(row, col))
        q_constant = 1.5
        evaporation_rate = 0.7

        ants = 256*2
        gens = 32

        shortest_path = 99999999

        # helper functions

        def roulette_select(current_node, nodes_to_check):

            n = np.random.uniform(0, 1)

            s = 0

            prob = []
            nodes = []

            for node in nodes_to_check:
                nodes.append(node)
                if(distance(current_node, node) != 0):
                    a = (1 / distance(current_node, node)) * \
                        pheromone(current_node, node)
                else:
                    a = pheromone(current_node, node)
                s += a
                prob.append(a)

            prob = np.array(prob, dtype='float64')
            prob = prob / s

            # cumulative_sum = []
            # total = 0

            # for i in range(len(prob)):
            #     total += prob[i]
            #     cumulative_sum[i] = total

            # for i in range(len(cumulative_sum)):
            #     if(n < cumulative_sum[i]):
            #         return_node = nodes[i]

            cumulative_sum = 0
            chosen = 0

            # for i in prob:
            #     if cumulative_sum < n:
            #         chosen = i
            #     cumulative_sum += i

            for i in range(len(prob)):
                if cumulative_sum < n:
                    chosen = i
                cumulative_sum += prob[i]

            # print('\n')
            # print("Probs are: ", prob)
            # print("Nodes: ", nodes)
            # print("Chosen: ", nodes[chosen])

            return nodes[chosen]

        def pheromone(p1, p2):
            pher = pheromones[p2[0]][p2[1]]
            return pher

        def distance(p1, p2):
            dist = grid[p2[0]][p2[1]]
            return dist

        def update_pheromones(paths):
            new_pheromones = (1 - evaporation_rate) * pheromones

            for hist, dist in paths:
                # print("hist: ", hist, " dist : ", dist)
                for node in hist:
                    i = node[0]
                    j = node[1]

                    if (dist == 0):
                        dist = 0.7

                    new_node_pher = new_pheromones[i][j] + (q_constant / dist)
                    new_pheromones[i][j] = new_node_pher

            # print(new_pheromones)
            return new_pheromones

        def update_nodes_to_check(node, path):

            # print('\n')
            # print("untc node: ", node)
            # print("untc path: ", path)

            i = node[0]
            j = node[1]

            # print("i ", i)
            # print("j ", j)

            new_nodes_to_check = set()

            if((i+1 < row) and ((i+1, j) not in path)):
                # print("south")
                new_nodes_to_check.add((i+1, j))
            if((i-1 >= 0) and ((i-1, j) not in path)):
                # print("north")
                new_nodes_to_check.add((i-1, j))
            if((j+1 < col) and ((i, j+1) not in path)):
                # print("east")
                new_nodes_to_check.add((i, j+1))
            if((j-1 >= 0) and ((i, j-1) not in path)):
                # print("west")
                new_nodes_to_check.add((i, j-1))

            # print("new nodes to check: ", new_nodes_to_check)
            return new_nodes_to_check

        def update_shortest_path(paths):

            # print(shortest_path)

            current_shortest = shortest_path

            for hist, dist in paths:
                # print("dist: ", dist)
                if dist < current_shortest:
                    current_shortest = dist

            return current_shortest

        for g in range(gens):

            paths = []

            for a in range(ants):

                current_node = (0, 0)
                current_distance = 0

                path = set()
                path.add(current_node)

                path_in_order = []
                path_in_order.append(current_node)

                nodes_to_check = set()

                nodes_to_check.add((1, 0))
                nodes_to_check.add((0, 1))

                while (len(nodes_to_check) != 0) and (current_node != end):

                    next_node = roulette_select(current_node, nodes_to_check)

                    current_distance += distance(current_node, next_node)

                    nodes_to_check = update_nodes_to_check(next_node, path)

                    current_node = next_node

                    path.add(next_node)

                    path_in_order.append(next_node)

                if(end in path):
                    paths.append([path_in_order, current_distance])

            pheromones = update_pheromones(paths)
            shortest_path = update_shortest_path(paths)

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

game = Game(13, 13)
grid_genrated = game.generateGrid()

grid = grid2

print('\n')
# compute distance with Dijkstra
begin_time = datetime.datetime.now()
distance = game.dijkstra(grid, (0, 0))
print("time   -   Dijkstra ", datetime.datetime.now() - begin_time)
print("distance - Dijkstra ", distance)

print('\n')
# compute distance with ant colony
begin_time = datetime.datetime.now()
distance3 = game.ant_colony(grid, (0, 0))
print("time   -   ant_colony ", datetime.datetime.now() - begin_time)
print("distance - ant_colony ", distance3)

print('\n')
# compute distance with BFS
begin_time = datetime.datetime.now()
distance2 = game.BFS(grid, (0, 0))
print("time   -   BFS ", datetime.datetime.now() - begin_time)
print("distance - BFS ", distance2)
