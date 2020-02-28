import pygame
import time
from lab2settings import *
import ast
from collections import *
from os import path
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
        self.directions = [vec(1, 0), vec(0, 1), vec(-1, 0), vec(0, -1)]
        self.running = False
        self.goal = goal
        self.start = start
        self.gridwidth = gridwidth
        self.gridheight = gridheight

        self.search = {}
        self.path = {}
        self.arrows = {}
        arrow_img = pygame.image.load('rightArrow.png').convert_alpha()
        arrow_img = pygame.transform.scale(arrow_img, (50, 50))
        for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            self.arrows[dir] = pygame.transform.rotate(arrow_img, vec(dir).angle_to(vec(1, 0)))

    def inBounds(self, node):
        return 0 <= node.x < self.gridwidth and 0 <= node.y < self.gridheight

    def passable(self, node):
        return node not in self.walls

    def findNeighbors(self, node):
        neighbors = []
        neighbors = [node + direction for direction in self.directions]
        neighbors = filter(self.inBounds, neighbors)
        neighbors = filter(self.passable, neighbors)
        neighbors = list(neighbors)
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
        while len(frontier) > 0:
            current = frontier.popleft()
            if current == self.start:
                break
            for nextNode in self.findNeighbors(current):
                if self.vec2int(nextNode) not in self.search:
                    frontier.append(self.vec2int(nextNode))
                    self.search[self.vec2int(nextNode)] = current - nextNode
        #creation of path
        if len(self.search) != 0 and self.goal != None:
            current = self.start#the current node
            while current != self.goal:
                current = current + self.search[self.vec2int(current)]
                #find the next direction in path
                self.path[self.vec2int(self.search[self.vec2int(current)])] = current

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
                        print("draw path!")
                        self.breadthFirstSearch(self.goal)
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

    def drawSearch(self):
        for node, dire in self.search.items():
                if dire:
                    x, y = node
                    x = x * tilesize + tilesize/2
                    y = y * tilesize + tilesize/2
                    img = self.arrows[self.vec2int(dire)]
                    r = img.get_rect(center=(x, y))
                    g.screen.blit(img, r)

    def drawPath(self): # draws the path from start to goal
        for dire, node in self.path.items(): #TODO this or path setting needs fixing...!!
                if dire:
                    x, y = node
                    x = x * tilesize + tilesize/2
                    y = y * tilesize + tilesize/2
                    img = self.arrows[dire]
                    r = img.get_rect(center=(x, y))
                    g.screen.blit(img, r)


    def run(self):
        self.running = True
        while self.running:
            self.clock.tick(FPS)
            self.handleEvents()
            pygame.display.set_caption("{:.2f}".format(self.clock.get_fps()))
            self.screen.fill(darkgray)
            self.drawGrid()
            self.draw()
            self.drawPath()
            #self.drawSearch()
            pygame.display.flip()



g = Grid(10,10)
g.run()
