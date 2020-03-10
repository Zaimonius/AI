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
vec = pygame.math.Vector2


class Grid: #https://www.youtube.com/watch?v=e3gbNOl4DiM
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
                mousePos = vec(pygame.mouse.get_pos()) // tilesize
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
            return int(self.weights.get(self.vec2int(nextNode), 0) + 14)
        elif current - nextNode != 0:
            return int(self.weights.get(self.vec2int(nextNode), 0) + 10)
        else:
            return 0

    def pathCost(self):
        prevNode = self.goal
        for node, dire in self.path.items():
            if dire:
                cost = cost + cost(prevNode, node)
        return cost

    def randomAStar(self):
        map = self.loadMap("map.txt")
        start, goal = self.randomPoints() #return [x1, y1], [x2, y2] from random points
        self.start = vec(start[0], start[1])
        self.goal = vec(goal[0], goal[1])
        self.aStarSearch(self.start)
        dist = self.pathCost()


    def randomList(self, amount):
        pass

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

    def heuristic(self, node1, node2):
        #manhattan
        return int(abs(node1.x - node2.x) + abs(node1.y - node2.y)) # times ten for weight scale

    def aStarSearch(self, startNode):
        frontier = deque()
        frontier.append(startNode)
        self.search = {}
        self.path = {}
        self.costSoFar = {}
        self.costSoFar[self.vec2int(startNode)] = 0
        while len(frontier) > 0:
            current = frontier.popleft()
            if current == self.goal:
                break
            for nextNode in self.findNeighbours(current):
                newCost = self.costSoFar[self.vec2int(current)] + self.cost(current, nextNode)
                if self.vec2int(nextNode) not in self.costSoFar or newCost < self.costSoFar[self.vec2int(nextNode)]:
                    self.costSoFar[self.vec2int(nextNode)] = newCost #TODO costs are wrong i think!
                    priority = self.heuristic(self.goal, nextNode)
                    frontier.insert(priority, nextNode)
                    self.search[self.vec2int(nextNode)] = current - nextNode
        #creation of path
        if len(self.search) != 0 and self.goal is not None:
            current = self.goal #the current node
            while current != self.start:
                #find the next direction in path
                self.path[self.vec2int(current)] = self.search[self.vec2int(current)]
                current = current + self.search[self.vec2int(current)]


class CustomDataSet(Dataset):
    def __init__(self,dataPoints):
        pass
        X, y = self.RMList(dataPoints)          
        #Store them in member variables.
        self.X = X
        self.y = y

class Net(nn.Module):
    def __init__(self, inputSize):
        super.__init__()
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




#trainDataset = CustomDataSet(20)




g = WeightedGrid(20, 20)
g.loadMap("Map3.txt")
g.run()


