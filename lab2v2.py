# pip3 install torch torchvision
# pip3 install pytmx
# pip3 install pygame
#jedi numpy

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
import time
import os.path as io
vec = pygame.Vector2



class Grid: #https:#www.youtube.com/watch?v=e3gbNOl4DiM
    def __init__(self, gridwidth=0, gridheight=0, window = False, start = None, goal = None):
        pygame.init()
        #screen
        self.window = window
        if window == True:
            self.screen = pygame.display.set_mode((gridwidth * tilesize, gridheight * tilesize))
        #clock
        self.clock = pygame.time.Clock()
        #pixel grid measuements
        self.width = tilesize * gridwidth
        self.height = tilesize * gridheight
        #standard grid measurements
        self.gridwidth = gridwidth
        self.gridheight = gridheight
        #the walls of the map
        self.walls = []
        # the directions that the agent can move in
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.diagonals = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        self.allDirections = self.diagonals + self.directions
        #running bool
        self.running = False
        #goal and start nodes
        self.goal = goal
        self.start = start
        #search list
        self.search = []
        self.path = []

    def Draw(self):
        #draws all the walls, the goal and the start
        if self.window == True:
            for wall in self.walls:
                rect = pygame.Rect(vec(wall[0], wall[1])* tilesize, (tilesize, tilesize))
                pygame.draw.rect(self.screen, lightgray, rect)
            if self.goal!= None:
                startRect = pygame.Rect(vec(self.goal[0], self.goal[1])* tilesize, (tilesize, tilesize))
                pygame.draw.rect(self.screen, red, startRect)
            if self.start != None:
                goalRect = pygame.Rect(vec(self.start[0], self.start[1]) * tilesize, (tilesize, tilesize))
                pygame.draw.rect(self.screen, magenta, goalRect)

    def DrawGrid(self):
        if self.window == True: 
            #Draws the grid and the background
            for x in range(0, self.width, tilesize):
                pygame.draw.line(self.screen, lightgray, (x, 0), (x, self.height))
            for y in range(0, self.width, tilesize):
                pygame.draw.line(self.screen, lightgray, (0, y), (self.width, y))

    def DrawSearch(self):
        if self.window == True:
            for node in self.search:
                x, y = node
                rect = pygame.Rect(vec(x, y)* tilesize, (tilesize, tilesize))
                pygame.draw.rect(self.screen, yellow, rect)
    
    def DrawPath(self):
        if self.window == True:
            for position in self.path:
                x, y = position
                rect = pygame.Rect(vec(x, y)* tilesize, (tilesize, tilesize))
                pygame.draw.rect(self.screen, cyan, rect)

    def RandomPoints(self):
        #picks random tiles for start and goal
        startOK = False
        goalOK = False
        #start
        while not startOK:
            x1 = random.randint(0, self.gridwidth-1)
            y1 = random.randint(0, self.gridheight-1)
            if (x1, y1) not in self.walls:
                startOK = True
        #goal
        while not goalOK:
            x2 = random.randint(0, self.gridwidth-1)
            y2 = random.randint(0, self.gridheight-1)
            if (x2, y2) not in self.walls and (x2, y2) is not (x1, y2):
                goalOK = True
        
        return (x1, y1), (x2, y2)

    def InBounds(self, node):
        return 0 <= node[0] < self.gridwidth and 0 <= node[0] < self.gridheight

    def Passable(self, node):
        return node not in self.walls
    
    def PassableDiagonal(self, node, diagonal):
        if (node[0] + diagonal[0], node[1] + 0) in self.walls or (node[0] + 0, node[1] + diagonal[1]) in self.walls:
            return False
        else:
            return True

    def FindNeighbours(self, node):
        neighbours = []
        neighbours = [(node[0] + direction[0], node[1] + direction[1]) for direction in self.directions]
        diagonalNeighbours = []
        for diagonal in self.diagonals:
            if self.PassableDiagonal(node, diagonal):
                diagonalNeighbours.append((node[0] + diagonal[0], node[1] + diagonal[1]))
        neighbours = neighbours + diagonalNeighbours
        neighbours = filter(self.InBounds, neighbours)
        neighbours = filter(self.Passable, neighbours)
        neighbors = list(neighbours)
        return neighbors

    def HandleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("BFS!")
                    self.BFS()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mousePosPix = pygame.mouse.get_pos()
                mousePos = (int(mousePosPix[0]/tilesize), int(mousePosPix[1]/tilesize))
                if event.button == 1:
                    if mousePos in self.walls:
                        g.walls.remove(mousePos)
                    else:
                        g.walls.append(mousePos)
                if event.button == 2:
                    self.start = mousePos
                if event.button == 3:
                    self.goal = mousePos

    def Run(self):
        self.running = True
        while self.running:
            self.clock.tick(FPS)
            self.HandleEvents()
            pygame.display.set_caption("{:.2f}".format(self.clock.get_fps()))
            self.screen.fill(darkgray)
            self.DrawGrid()
            self.DrawSearch()
            self.DrawPath()
            self.Draw()
            pygame.display.flip()

    def BFS(self):
        startNode = Node(None, self.start)
        endNode = Node(None, self.goal)
        #frontier queue
        frontier = deque()
        frontier.append(startNode)
        #reset the search and path
        self.search = []
        self.path = []
        #the search
        while len(frontier) > 0:
            #pop the first in the frontier queue
            current = frontier.popleft()
            #if the current nodes posistion is the end nodes position we found the goal!
            if current.position == endNode.position:
                endNode.parent = current
                break
            #if not found add all the neighbours of the current node to frontier if they have not been added
            for nextNodePos in self.FindNeighbours(current.position):
                if nextNodePos not in self.search:
                    #make the next node
                    nextNode = Node(current, nextNodePos)
                    #set the next nodes parent to the current node
                    nextNode.parent = current
                    #add the neighbour to the frontier
                    frontier.append(nextNode)
                    #add the current node to the search because it has been discovered
                    self.search.append(nextNode.position)
        #creation of path
        current = endNode
        #while the current tile is not starting tile
        while current is not startNode:
            #insert the current position into the path
            self.path.insert(0, current.position)
            #then go to the next node in the path
            current = current.parent
        return self.path

    def loadMap(self, filePath):
        with open(filePath) as f:
            lineList = f.readlines()
            gridwidth = len(lineList[0]) - 1
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
                            self.walls.append((i, i2))
                        elif letter == "S":
                            self.start = (i, i2)
                        elif letter == "G":
                            self.goal = (i, i2)
                    i2 = i2 + 1
                i2 = 0
                i = i + 1

    def mapList(self, h=None, w=None, newStart=None):
        if h == None or w == None:
            h = self.gridheight
            w = self.gridwidth
        #only for training neural network
        listMap = []
        for i in range(h):
            listMap = listMap + [[0]*w]
        if newStart == None:
            listMap[int(self.start[0])][int(self.start[1])] = 2
        else:
            listMap[int(newStart[0])][int(newStart[1])] = 2
        listMap[int(self.goal[0])][int(self.goal[1])] = 2
        return listMap

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        #g is the cost of the path from the start node to this node
        self.g = 0
        #h is the heuristic function cost that estimates the cheapest path from the node to the goal
        self.h = 0
        #f are the h and g value combined into one value
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class WeightedGrid(Grid):
    def __init__(self, gridwidth=0, gridheight=0, window = False, start = None, goal = None):
        super().__init__(gridwidth, gridheight, window, start, goal)
        self.h = HeuristicNet(gridwidth, gridheight, "neural10000.txt")

    def HandleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    start = time.time_ns()
                    print("BFS!")
                    self.BFS()
                    end = time.time_ns()
                    print("Search time: " + str(start - end))
                    print("Search length: " + str(len(self.search)))
                    print("Path length: " + str(len(self.path)))
                if event.key == pygame.K_w:
                    print("A*")
                    start = time.time_ns()
                    self.AStar()
                    end = time.time_ns()
                    print("Search time: " + str(start - end))
                    print("Search length: " + str(len(self.search)))
                    print("Path length: " + str(len(self.path)))
                if event.key == pygame.K_e:
                    print("A* neural")
                    start = time.time_ns()
                    self.AStarNet()
                    end = time.time_ns()
                    print("Search time: " + str(start - end))
                    print("Search length: " + str(len(self.search)))
                    print("Path length: " + str(len(self.path)))
                if event.key == pygame.K_r:
                    self.start, self.goal = self.RandomPoints()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mousePosPix = pygame.mouse.get_pos()
                mousePos = (int(mousePosPix[0]/tilesize), int(mousePosPix[1]/tilesize))
                if event.button == 1:
                    if mousePos in self.walls:
                        g.walls.remove(mousePos)
                    else:
                        g.walls.append(mousePos)
                if event.button == 2:
                    self.start = mousePos
                if event.button == 3:
                    self.goal = mousePos

    def Heuristic(self, node1, node2=None):
        if node2 is None:
            node2 = self.goal
        #manhattan + chebyshev
        dx = abs(node1[0] - node2[0])
        dy = abs(node1[1] - node2[1])
        D = 1
        D2 = math.sqrt(2) #chebyshev   #math.sqrt(2)  #octile
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def Cost(self, pos1, pos2):
        #calculates the cost to move between two positions
        if pos1[0] - pos2[0] > 1 or pos1[1] - pos2[1] > 1:
            pass
        direction = (pos1[0]-pos2[0], pos1[1], pos2[1])
        if direction in self.diagonals:
            return 1.4
        else:
            return 1

    def AStar(self):
        #Create start and end nodes
        startNode = Node(None, self.start)
        startNode.g = startNode.h = startNode.f = 0
        endNode = Node(None, self.goal)
        endNode.g = endNode.h = endNode.f = 0

        #create open and closed list
        openList = [startNode]
        closedList = []

        #reset the path and search lists
        self.path = []
        self.search = []

        #create the loop
        while len(openList) > 0:
            #reset index and node
            current = openList[0]
            currentIndex = 0
            #loop for the next index and node with the least f value
            for index, item in enumerate(openList):
                if item.f < current.f:
                    current = item
                    currentIndex = index
            #remove the current from the open list and append it to the closed list
            openList.pop(currentIndex)
            closedList.append(current)

            if current.position == endNode.position:
                self.path = []
                while current is not None:
                    self.path.append(current.position) #prepend?
                    current = current.parent
                return
            
            #generate neigbours
            for newPos in self.FindNeighbours(current.position): #TODO check if in serach here
                #check if neighbour already in the closed list
                if newPos in self.search:
                    pass
                else:
                    #create new node
                    newNode = Node(current, newPos)
                    #calculate the cost to get from this node to the neighbour
                    cost = current.g + self.Cost(current.position, newNode.position)
                    #if the neighbour is already in the open list, dont add it
                    if newNode in openList:
                        if newNode.g > cost: #if there is already a cost, compare which is best
                            newNode.g = cost
                    else:
                        #else add the neighbour
                        newNode.g = cost
                        openList.append(newNode)
                        self.search.append(newNode.position)
                    #create all the values for h and f
                    newNode.h = self.Heuristic(newNode.position, endNode.position)
                    newNode.f = newNode.g + newNode.h

    def AStarNet(self):
        #Create start and end nodes
        startNode = Node(None, self.start)
        startNode.g = startNode.h = startNode.f = 0
        endNode = Node(None, self.goal)
        endNode.g = endNode.h = endNode.f = 0

        #create open and closed list
        openList = [startNode]
        closedList = []

        #reset the path and search lists
        self.path = []
        self.search = []

        #create the loop
        while len(openList) > 0:
            #reset index and node
            current = openList[0]
            currentIndex = 0
            #loop for the next index and node with the least f value
            for index, item in enumerate(openList):
                if item.f < current.f:
                    current = item
                    currentIndex = index
            #remove the current from the open list and append it to the closed list
            openList.pop(currentIndex)
            closedList.append(current)

            if current.position == endNode.position:
                self.path = []
                while current is not None:
                    self.path.append(current.position) #prepend?
                    current = current.parent
                return
            
            #generate neigbours
            for newPos in self.FindNeighbours(current.position): #TODO check if in serach here
                #check if neighbour already in the closed list
                if newPos in self.search:
                    pass
                else:
                    #create new node
                    newNode = Node(current, newPos)
                    #calculate the cost to get from this node to the neighbour
                    cost = current.g + self.Cost(current.position, newNode.position)
                    #if the neighbour is already in the open list, dont add it
                    if newNode in openList:
                        if newNode.g > cost: #if there is already a cost, compare which is best
                            newNode.g = cost
                    else:
                        #else add the neighbour
                        newNode.g = cost
                        openList.append(newNode)
                        self.search.append(newNode.position)
                    #create all the values for h and f
                    newNode.h = float(self.h.neuralHeuristic(self.mapList(self.h.gridheight, self.h.gridwidth, newNode.position)))
                    newNode.f = newNode.g + newNode.h

                    #print("regular: " + str(self.Heuristic(newNode.position, endNode.position)) + " net: " + str(newNode.h))
    
    def PathCost(self):
        prev = self.start
        cost = 0
        for node in self.path:
            cost = cost + self.Cost(prev, node)
            prev = node
        return cost
    
    def RandomAStar(self, grid):
        start, goal = grid.RandomPoints() #return (x1, y1), (x2, y2) from random points
        grid.start = start
        grid.goal = goal
        grid.AStar()
        dist = grid.PathCost()
        theMap = grid.mapList()
        return theMap, dist
    
    def RandomAStarList(self, size, gridsize, grid):
        x = []
        y = []
        for i in range(size):
            a, b = self.RandomAStar(grid)
            x = x + [a]
            y = y + [b]
        return x, y

class CustomDataSet(Dataset):
    def __init__(self, dataPoints, gridwidth, gridheight):
        self.g = WeightedGrid(gridwidth, gridheight, False)
        X, y = self.g.RandomAStarList(dataPoints, size, self.g)
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self, index):
        return torch.FloatTensor(self.X[index]),self.y[index]

class Net(nn.Module):
    def __init__(self, inputheight, inputwidth):
        super().__init__()
        self.fc1 = nn.Linear(inputheight * inputwidth, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 100)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class HeuristicNet:
    def __init__(self, gridwidth=28, gridheight=28, filePath=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if filePath is not None and io.exists(filePath):
            self.load(filePath)
            self.net.to(self.device)
            self.net.eval()
        else:
            self.gridwidth = gridwidth
            self.gridheight = gridheight
            self.net = Net(gridwidth, gridheight)

    def train(self, dataPoints, epochs, savePath=None):
        #create the neural net
        self.net.train()

        #decide loss function and optmizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)

        #Teach the NN
        for datasets in range(epochs):
            print("Epoch #", datasets)
            train = CustomDataSet(dataPoints, self.gridwidth, self.gridheight)
            trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
            for data in trainset:  # `data` is a batch of data
                X, y = data  # X is the batch of features, y is the batch of targets.
                self.net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                output = self.net(X.view(-1,  self.gridwidth* self.gridheight))  # pass in the reshaped batch 
                loss = F.nll_loss(output, y)  # calc and grab the loss value
                loss.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients

        # Neural network procentage test
        self.net.eval()
        test = CustomDataSet(dataPoints, self.gridwidth, self.gridheight)
        testset = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testset:
                X, y = data
                output = self.net(X.view(-1,self.gridwidth* self.gridheight))
                for idx, i in enumerate(output):
                    print(torch.argmax(i), y[idx])
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        print("Accuracy: ", round((correct/total)*100, 3))
        self.save(savePath)

    def save(self, savePath):
        #save the neural network
        if savePath is not None:
            obj = {'dict': self.net.state_dict(), 'gridwidth': self.gridwidth, 'gridheight': self.gridheight}
            torch.save(obj, savePath) #save input size and dictionary
            print(savePath + " saved")


    def load(self, loadPath):
        #loading neural net
        #TODO: fix this with width x height
        print(loadPath + " loaded")
        load_data = torch.load(loadPath)
        self.net = Net(load_data['gridwidth'], load_data['gridheight']) #inputsize
        self.net.load_state_dict(load_data['dict']) # dictionary
        self.gridwidth = load_data['gridwidth']
        self.gridheight = load_data['gridheight']

    def neuralHeuristic(self, mapList):
        # g = WeightedGrid(self.net.inputSize, self.net.inputSize)
        # g.goal = goal
        # g.start = start
        X = torch.Tensor(mapList)
        X = X.to(self.device)
        output = self.net(X.view(-1, self.gridheight*self.gridwidth))
        if torch.cuda.is_available():
            output = output.cuda()
        else:
            output = output.cpu()
        return int(torch.argmax(output))


print("cuda: " + str(torch.cuda.is_available()))


# g = WeightedGrid(20, 20, True)
# g.loadMap("Map1.txt")
# g.Run()

#neural net inputs
size = 28
dataPoints = 20
savePath = "neural100.txt"


# #train 100 epoch net
h = HeuristicNet(30,30)
h.train(dataPoints, 100, "neural100.txt")

# #train 1000 epoch net
h = HeuristicNet(30,30)
h.train(dataPoints, 1000, "neural1000.txt")

# #train 10000 net
h = HeuristicNet(30,30)
h.train(dataPoints, 10000, "neural10000.txt")

# #train 100000 net
h = HeuristicNet(30,30)
h.train(dataPoints, 100000, "neural100000.txt")