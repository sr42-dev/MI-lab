"""
You can create any other helper funtions.
Do not modify the given functions
"""

class Graph :

    def __init__(self, cost, start, goals, heuristic = []):

        self.cost = cost # 2D list
        self.heuristic = heuristic # 1D list
        self.start = start # integer
        self.goals = goals # 1D list
        self.pathCost = 0

    def aStarRecurse(self, node, path):
        
        path.append(node)
        possibleNodes = []
        for i in range(1, len(self.cost[node])):
            tuple = (i, self.cost[node][i]) # node number, cost to traverse
            if ((self.cost[node][i] != 0) and (self.cost[node][i] != -1)):
                tuple = (i, self.cost[node][i] + self.heuristic[i]) 
                possibleNodes.append(tuple)

        print(possibleNodes)
        nextNode = min(possibleNodes, key = lambda t: t[1])[0]
        print(nextNode)

        if nextNode not in self.goals :
            self.aStarRecurse(nextNode, path)
        else :
            path.append(nextNode)
            
    def aStar(self) :

        path = []

        self.aStarRecurse(self.start, path)
        #print(path)

        return path

    def dfsRecurse(self, node, path) :

        path.append(node)
        possibleNodes = []
        for i in range(1, len(self.cost[node])):
            tuple = (i, self.cost[node][i]) # node number, cost to traverse
            if ((self.cost[node][i] != 0) and (self.cost[node][i] != -1)):
                possibleNodes.append(tuple)

        nextNode = min(possibleNodes, key = lambda t: t[1])[0]

        if nextNode not in self.goals :
            self.dfsRecurse(nextNode, path)
        else :
            path.append(nextNode)

    def dfs(self) :

        path = []

        self.dfsRecurse(self.start, path)

        return path

def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """

    graph = Graph(cost, start_point, goals, heuristic)
    path = graph.aStar()

    # TODO
    return path

def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    
    graph = Graph(cost, start_point, goals)
    path = graph.dfs()

    # TODO
    return path
