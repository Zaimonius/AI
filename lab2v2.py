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
        #screen
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

    def draw(self):
        #draws all the walls, the goal and the start
        for wall in self.walls:
            rect = pygame.Rect(wall * tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, lightgray, rect)
        if self.goal!= None:
            startRect = pygame.Rect(vec(self.goal[0], self.goal[1])* tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, red, startRect)
        if self.start != None:
            goalRect = pygame.Rect(vec(self.start[0], self.start[1]) * tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, magenta, goalRect)

    def drawGrid(self):
        #Draws the grid and the background
        for x in range(0, self.width, tilesize):
            pygame.draw.line(self.screen, lightgray, (x, 0), (x, self.height))
        for y in range(0, self.width, tilesize):
            pygame.draw.line(self.screen, lightgray, (0, y), (self.width, y))

    def drawSearch(self):
        for node in self.search:
            x, y = node.posistion
            x = x * tilesize + tilesize/2 #4
            y = y * tilesize + tilesize/2 #4
            r = pygame.Rect(x, y, tilesize/2, tilesize/2)
            surface = pygame.Surface((tilesize/2, tilesize/2))
            surface.fill(yellow)
            g.screen.blit(surface, r)
    
    def drawPath(self):
        for position in self.path:
            x, y = position
            x = x * tilesize + tilesize/2 #4
            y = y * tilesize + tilesize/2 #4
            r = pygame.Rect(x, y, tilesize/2, tilesize/2)
            surface = pygame.Surface((tilesize/2, tilesize/2))
            surface.fill(cyan)
            g.screen.blit(surface, r)


    def randomPoints(self):
        #picks random tiles for start and goal
        startOK = False
        goalOK = False
        #start
        while not startOK:
            x1 = random.randint(0, self.gridwidth)
            y1 = random.randint(0, self.gridheight)
            if (x1, y1) not in self.walls:
                startOK = True
        #goal
        while not goalOK:
            x2 = random.randint(0, self.gridwidth)
            y2 = random.randint(0, self.gridheight)
            if (x2, y2) not in self.walls:
                goalOK = True
        return (x1, y1), (x2, y2)

    def inBounds(self, node):
        return 0 <= node[0] < self.gridwidth and 0 <= node[0] < self.gridheight

    def passable(self, node):
        return node not in self.walls
    
    def passableDiagonal(self, node, diagonal):
        if (node[0] + diagonal[0], node[1] + 0) in self.walls or (node[0] + 0, node[1] + diagonal[1]) in self.walls:
            return False
        else:
            return True

    def findNeighbours(self, node):
        neighbours = []
        neighbours = [(node[0] + direction[0], node[1] + direction[1]) for direction in self.directions]
        diagonalNeighbours = []
        for diagonal in self.diagonals:
            if self.passableDiagonal(node, diagonal):
                diagonalNeighbours.append((node[0] + diagonal[0], node[1] + diagonal[1]))
        neighbours = neighbours + diagonalNeighbours
        neighbours = filter(self.inBounds, neighbours)
        neighbours = filter(self.passable, neighbours)
        neighbors = list(neighbours)
        return neighbors

    def breadthFirstSearch(self):
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
            for nextNodePos in self.findNeighbours(current.position):
                if nextNodePos not in self.search:
                    #make the next node
                    nextNode = Node(current, nextNodePos)
                    #set the next nodes parent to the current node
                    nextNode.parent = current
                    #add the neighbour to the frontier
                    frontier.append(nextNode)
                    #add the current node to the search because it has been discovered
                    self.search.append(current.position)
        #creation of path
        current = endNode
        #while the current tile is not starting tile
        while current is not startNode:
            #insert the current position into the path
            self.path.insert(0, current.position)
            #then go to the next node in the path
            current = current.parent
        return self.path


    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    print("BFS!")
                    self.breadthFirstSearch()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mousePosPix = pygame.mouse.get_pos()
                mousePos = (int(mousePosPix[0]/tilesize), int(mousePosPix[1]/tilesize))
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

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class WeigthedGrid:
    pass


g = Grid(20, 20)
g.run()
    