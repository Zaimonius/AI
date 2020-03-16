# pip3 install torch torchvision
# pip3 install pytmx
# pip3 install pygame

from lab2settings import *
from collections import *
import pygame
import pygame.locals
import ast
from os import path
import heapq
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import random
import math
vec = pygame.math.Vector2

class Grid: #https:#www.youtube.com/watch?v=e3gbNOl4DiM
    def __init__(self, gridwidth, gridheight, start = None, goal = None):
        pygame.init()
        self.screen = pygame.display.set_mode((gridwidth * tilesize, gridheight * tilesize))
        self.clock = pygame.time.Clock()
        #the maps width
        self.width = tilesize * gridwidth
        #the maps height
        self.height = tilesize * gridheight
        #the walls of the map
        self.walls = []
        # the directions that the agent can move in
        self.gridwidth = gridwidth
        self.gridheight = gridheight
        self.directions = [vec(1, 0), vec(0, 1), vec(-1, 0), vec(0, -1)]
        self.diagonals = [vec(1, 1), vec(-1, 1), vec(-1, -1), vec(1, -1)]
        self.running = False
        self.goal = goal
        self.start = start
        self.search = {}
        self.path = {}
        self.arrows = {}
        self.costSoFar = {}
        arrow_img = pygame.image.load('rightArrow.png').convert_alpha()
        arrow_img = pygame.transform.scale(arrow_img, (50, 50))
        for dire in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]:
            self.arrows[dire] = pygame.transform.rotate(arrow_img, vec(dire).angle_to(vec(1, 0)))

    def inBounds(self, node):
        return 0 <= node.x < self.gridwidth and 0 <= node.y < self.gridheight

    def passable(self, node):
        return node not in self.walls
    
    def passableDiagonal(self, node, diagonal):
        if node + vec(diagonal.x, 0) in self.walls or node + vec(0, diagonal.y) in self.walls:
            return False
        else:
            return True

    def findNeighbours(self, node):
        neighbours = []
        neighbours = [node + direction for direction in self.directions]
        diagonalNeighbours = []
        for diagonal in self.diagonals:
            if self.passableDiagonal(node, diagonal):
                diagonalNeighbours.append(node + diagonal)
        neighbours = neighbours + diagonalNeighbours
        neighbours = filter(self.inBounds, neighbours)
        neighbours = filter(self.passable, neighbours)
        neighbors = list(neighbours)
        return neighbors

    def draw(self):
        for wall in self.walls:
            rect = pygame.Rect(wall * tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, lightgray, rect)
        if self.goal!= None:
            startRect = pygame.Rect(self.goal* tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, red, startRect)
        if self.start != None:
            goalRect = pygame.Rect(self.start * tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, magenta, goalRect)

    def drawGrid(self):
        for x in range(0, self.width, tilesize):
            pygame.draw.line(self.screen, lightgray, (x, 0), (x, self.height))
        for y in range(0, self.width, tilesize):
            pygame.draw.line(self.screen, lightgray, (0, y), (self.width, y))

    def loadDrawn(self):
        f = open("drawnmap.txt", "r")
        wallList = f.readline()
        if wallList != "":
            wallPairs = ast.literal_eval(wallList)
            for wall in wallPairs:
                self.walls.append(vec(wall[0], wall[1]))
        f.close()

    @staticmethod
    def vec2int(vect):
        return (int(vect.x), int(vect.y))

    def breadthFirstSearch(self, startNode):
        #the search
        frontier = deque()
        frontier.append(startNode)
        self.search = {}
        self.path = {}
        while len(frontier) > 0:
            current = frontier.popleft()
            if current == self.goal:
                break
            for nextNode in self.findNeighbours(current):
                if self.vec2int(nextNode) not in self.search:
                    frontier.append(self.vec2int(nextNode))
                    self.search[self.vec2int(nextNode)] = current - nextNode
        #creation of path
        if len(self.search) != 0 and self.goal is not None:
            current = self.goal #the current node
            while current != self.start:
                current = current + self.search[self.vec2int(current)]
                #find the next direction in path
                self.path[self.vec2int(current)] = self.search[self.vec2int(current)]

    def depthFirstSearch(self, startNode): #this is not a good depth first
        #the search
        visited, stack = set(), [startNode]
        self.search = {}
        self.path = {}
        while len(stack) > 0:
            current = stack.pop()
            if current == self.goal:
                break
            for nextNode in self.findNeighbours(current):
                if self.vec2int(nextNode) not in self.search:
                    stack.append(self.vec2int(nextNode))
                    self.search[self.vec2int(nextNode)] = current - nextNode
        #creation of path
        if len(self.search) != 0 and self.goal is not None:
            current = self.goal #the current node
            while current != self.start:
                current = current + self.search[self.vec2int(current)]
                #find the next direction in path
                self.path[self.vec2int(current)] = self.search[self.vec2int(current)]

    def drawSearch(self):
        for node, dire in self.search.items():
            if dire:
                x, y = node
                x = x * tilesize + tilesize/2 #4
                y = y * tilesize + tilesize/2 #4
                img = self.arrows[self.vec2int(dire)]
                r = img.get_rect(center=(x, y))
                g.screen.blit(img, r)
                #r = pygame.Rect(x, y, tilesize/2, tilesize/2)
                #surface = pygame.Surface((tilesize/2, tilesize/2))
                #surface.fill(yellow)
                #g.screen.blit(surface, r)

    def drawPath(self): # draws the path from start to goal
        for node, dire in self.path.items():
            if dire:
                x, y = node
                x = x * tilesize + tilesize/2
                y = y * tilesize + tilesize/2
                img = self.arrows[self.vec2int(dire)]
                r = img.get_rect(center=(x, y))
                g.screen.blit(img, r)

    def randomPoints(self):
        startOK = False
        goalOK = False
        #start
        while not startOK:
            x1 = random.randint(0, self.gridwidth)
            y1 = random.randint(0, self.gridheight)
            if vec(x1, y1) not in self.walls:
                startOK = True
        #goal
        while not goalOK:
            x2 = random.randint(0, self.gridwidth)
            y2 = random.randint(0, self.gridheight)
            if vec(x2, y2) not in self.walls:
                goalOK = True
        return [x1, y1], [x2, y2]

    def loadMap(self, filePath):
        with open(filePath) as f:
            lineList = f.readlines()
            gridwidth = len(lineList[0]) -1
            gridheight = len(lineList)
            self.screen = pygame.display.set_mode((gridwidth * tilesize, gridheight * tilesize))
            self.width = tilesize * gridwidth
            self.height = tilesize * gridheight
            self.gridwidth = gridwidth
            self.gridheight = gridheight
            i = 0
            i2 = 0
            for line in lineList:
                for letter in line:
                    if letter != 0:
                        if letter == "X":
                            self.walls.append(vec(i, i2))
                        elif letter == "S":
                            self.start = vec(i, i2)
                        elif letter == "G":
                            self.goal = vec(i, i2)
                    i2 = i2 + 1
                i2 = 0
                i = i + 1

    def mapToList(self):
        #only for training neural network
        listMap = []
        for i in range(self.gridheight):
            listMap = listMap + [[0]*self.gridwidth]

        listMap[self.start.x][self.start.y] = 2
        listMap[self.goal.x][self.goal.y] = 2

class PriorityQueue:
    def __init__(self):
        self.nodes = []

    def put(self, node, cost):
        heapq.heappush(self.nodes, (cost, node))

    def get(self):
        return heapq.heappop(self.nodes)[0]

    def empty(self):
        return len(self.nodes) == 0

class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class WeightedGrid(Grid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_q: #save map
                    f = open("drawnmap.txt", "w")
                    wallList = []
                    for wall in self.walls:
                        wallList.append([wall.x, wall.y])
                    f.write(str(wallList))
                    f.close()
                    print("map saved")
                if event.key == pygame.K_w:
                    self.loadDrawn()
                    print("map loaded")
                if event.key == pygame.K_e:
                    print("breadth first!")
                    self.breadthFirstSearch(self.start)
                if event.key == pygame.K_r:
                    print("diJkstras!")
                    self.dijkstraSearch(self.start)
                if event.key == pygame.K_t:
                    print("ASTAR!")
                    self.aStarSearch(self.start)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mousePos = vec(pygame.mouse.getPos()) # tilesize
                if event.button == 1:
                    if mousePos in self.walls:
                        g.walls.remove(mousePos)
                    else:
                        g.walls.append(mousePos)
                if event.button == 2:
                    self.start = mousePos
                if event.button == 3:
                    self.goal = mousePos

    def run(self):
        self.running = True
        while self.running:
            self.clock.tick(FPS)
            self.handleEvents()
            pygame.display.set_caption("{:.2f}".format(self.clock.get_fps()))
            self.screen.fill(darkgray)
            self.drawGrid()
            self.drawSearch()
            self.draw()
            #self.drawPath()
            pygame.display.flip()

    def cost(self, current, nextNode):
        if current - nextNode in self.diagonals:
            return 14
        elif current - nextNode != 0:
            return 10
        else:
            return 0

    def pathCost(self):
        prevNode = self.goal
        cost = 0
        for node, dire in self.path.items():
            if dire:
                cost = cost + self.cost(prevNode, node)
        return cost

    def randomAStar(self):
        map = self.loadMap("map.txt")
        start, goal = self.randomPoints() #return [x1, y1], [x2, y2] from random points
        self.start = vec(start[0], start[1])
        self.goal = vec(goal[0], goal[1])
        self.aStarSearch(self.start)
        dist = self.pathCost()
        theMap = self.mapToList()
        return theMap, dist

    def randomList(self, amount):
        x = []
        y = []
        for i in range(amount):
            a, b = self.randomAStar()
            x = x + [a]
            y = y + [b]
        return (x, y)

    def dijkstraSearch(self, startNode): #  same as breadth first but with weights
        frontier = PriorityQueue()
        frontier.put(self.vec2int(startNode), 0)
        self.search = {}
        self.path = {}
        self.costSoFar = {}
        self.costSoFar[self.vec2int(startNode)] = 0
        while frontier.empty():
            current = frontier.get()
            if current == self.goal:
                break
            for nextNode in self.findNeighbours(current):
                newCost = self.costSoFar[self.vec2int(current)] + self.cost(current, nextNode)
                if self.vec2int(nextNode) not in self.costSoFar or newCost < self.costSoFar[self.vec2int(nextNode)]:
                    self.costSoFar[self.vec2int(nextNode)] = newCost
                    priority = newCost
                    frontier.put(self.vec2int(nextNode), priority)
                    self.search[self.vec2int(nextNode)] = current - nextNode
        #creation of path
        if len(self.search) != 0 and self.goal is not None:
            current = self.goal #the current node
            while current != self.start:
                #find the next direction in path
                self.path[self.vec2int(current)] = self.search[self.vec2int(current)]
                current = current + self.search[self.vec2int(current)]

    def heuristic(self, node1, node2=None):
        if node2 == None:
            node2 = self.goal
        #diagonal manhattan distance
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        D = 1
        D2 = math.sqrt(2)
        return int(D * (dx + dy) + (D2 - 2 * D) * min(dx, dy))

    def reconstructPath(self, cameFrom, current):
        totalPath = [current]
        while self.vec2int(current) in cameFrom.keys():
            current = cameFrom[self.vec2int(current)]
            totalPath.insert(0, current)
        self.path = totalPath
        return totalPath

    def aStarSearch(self, startNode):
        #discovered nodes
        openList = deque()
        openList.append(startNode)
        closedList = []
        #for node n cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known
        cameFrom =  {}
        # For node n, gScore[n] is the cost of the cheapest path from start to n currently known
        gScore = {}
        gScore[self.vec2int(startNode)] = 0

        # For node n, fScore[n] = gScore[n] + h(n). fScore[n] represents our current best guess as to
        # how short a path from start to finish can be if it goes through n
        fScore = {}
        fScore[self.vec2int(startNode)] = self.heuristic(startNode)
        while len(openList):
            current = openList.popleft() # the node in open list having the lowest fScore value
            closedList.append(current)
            if current == self.goal:
                return self.reconstructPath(cameFrom, current)
            for neighbor in self.findNeighbours(current):
                # d(current,neighbor) is the weight of the edge from current to neighbor
                # tentative_gScore is the distance from start to the neighbor through current
                tentativeGScore = gScore[self.vec2int(current)] + self.cost(current, neighbor)
                if tentativeGScore < gScore[self.vec2int(neighbor)]:
                    # This path to neighbor is better than any previous one. Record it!
                    cameFrom[self.vec2int(neighbor)] = current
                    gScore[self.vec2int(neighbor)] = tentativeGScore
                    fScore[self.vec2int(neighbor)] = gScore[self.vec2int(neighbor)] + self.heuristic(neighbor)
                    if neighbor not in closedList:
                        self.search[self.vec2int(current)] = current - neighbor
                        openList.append(neighbor)
        print("done")
        return

    def AStar(self, startNode):    
        # Create start and end node
        opa = self.vec2int(self.start)
        end = self.vec2int(self.goal)
        startNode = Node(None, start1)
        startNode.g = startNode.h = startNode.f = 0
        endNode = Node(None, end1)
        endNode.g = endNode.h = endNode.f = 0
        
        # Initialize both open and closed list
        openList = []
        closedList = []

        # Add the start node
        openList.append(startNode)

        # Loop until you find the end
        while len(openList) > 0:

            # Get the current node
            currentNode = openList[0]
            currentIndex = 0
            for index, item in enumerate(openList):
                if item.f < currentNode.f:
                    currentNode = item
                    currentIndex = index

            # Pop current off open list, add to closed list
            openList.pop(currentIndex)
            closedList.append(currentNode)

            # Found the goal
            if currentNode == endNode:
                path = []
                current = currentNode
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path

            # Generate children
            children = []
            for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

                # Get node position
                nodePosition = (currentNode.position[0] + newPosition[0], currentNode.position[1] + newPosition[1])

                # Make sure within range
                if nodePosition[0] > (len(maze) - 1) or nodePosition[0] < 0 or nodePosition[1] > (len(maze[len(maze)-1]) -1) or nodePosition[1] < 0:
                    continue

                # Make sure walkable terrain
                if maze[nodePosition[0]][nodePosition[1]] != 0:
                    continue

                # Create new node
                newNode = Node(currentNode, nodePosition)

                # Append
                children.append(newNode)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closedChild in closedList:
                    if child == closedChild:
                        continue

                # Create the f, g, and h values
                child.g = currentNode.g + 1
                child.h = ((child.position[0] - endNode.position[0]) ** 2) + ((child.position[1] - endNode.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for openNode in openList:
                    if child == openNode and child.g > openNode.g:
                        continue

                # Add the child to the open list
                openList.append(child)

class CustomDataSet(Dataset):
    def __init__(self, dataPoints):
        self.g = WeightedGrid(dataPoints, dataPoints)
        X, y = self.g.randomList(dataPoints)
        self.X = X
        self.y = y

class Net(nn.Module):
    def __init__(self, inputSize):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 100)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def __len__(self):
        return len(self.y)
   
    def __getitem__(self, index):
        return torch.FloatTensor(self.X[index]),self.y[index]

#trainDataset = CustomDataSet(28)

#textPp = "B==========D"
g = WeightedGrid(20, 20)
g.loadMap("Map1.txt")
g.run()

#net = Net(28)

#Create train and test data
#train = CustomDataSet(20)
#test = CustomDataSet(20)
#make them sets
#trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
#testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

#decide loss function and optmizer
#loss_function = nn.CrossEntropyLoss()
#optimizer = optim.Adam(net.parameters(), lr=0.001)

#Teach the NN
#for datasets in range(200): #10000
#    print("Epoch #", datasets)
#    train = CustomDataset(200)
#    trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
#    for epoch in range(10): # 3 full passes over the data
#        for data in trainset:  # `data` is a batch of data
#            X, y = data  # X is the batch of features, y is the batch of targets.
#            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
#            output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)
#            loss = F.nll_loss(output, y)  # calc and grab the loss value
#            loss.backward()  # apply this loss backwards thru the network's parameters
#            optimizer.step()  # attempt to optimize weights to account for loss/gradients


# Test the NN
#net.eval() # needed?
#correct = 0
#total = 0
#with torch.no_grad():
#    for data in testset:
#        X, y = data
#        output = net(X.view(-1,784))
#        #print(output)
#        for idx, i in enumerate(output):
#            print(torch.argmax(i), y[idx])
#            if torch.argmax(i) == y[idx]:
#                correct += 1
#            total += 1

#print("Accuracy: ", round((correct/total)*100, 3))

## Save and load a model parameters:
##torch.save(net.state_dict(), PATH)
##
##net = Net()   #TheModelClass(*args, **kwargs)
##net.load_state_dict(torch.load(PATH))
##net.eval()

