import pygame
import time
from lab2settings import *
import ast
vec = pygame.math.Vector2


class Grid: #https://www.youtube.com/watch?v=e3gbNOl4DiM
    def __init__(self, gridwidth, gridheight):
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
        self.directions = [vec(1, 0), vec(0, -1), vec(-1, 0), vec(0, -1)]
        self.running = False

    
    def inBounds(self, node):
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def passable(self, node):
        return node not in self.walls

    def findNeighbors(self, node):
        neighbors = []
        for direction in self.directions:
            neighbors.append(node + direction)
        for neighbor in neighbors:
            if not self.passable(neighbor) and not self.inBounds(neighbor):
                neighbors.remove(neighbor)
        print(neighbors)

    def draw(self):
        for wall in self.walls:
            rect = pygame.Rect(wall * tilesize, (tilesize, tilesize))
            pygame.draw.rect(self.screen, lightgray, rect)

    def drawGrid(self):
        for x in range(0, self.width, tilesize):
            pygame.draw.line(self.screen, lightgray, (x, 0), (x, self.height))
        for y in range(0, self.width, tilesize):
            pygame.draw.line(self.screen, lightgray, (0, y), (self.width, y))
    
    def loadDrawn(self):
        f = open("drawnmap.txt","r")
        wallList = f.readline()
        if wallList != "":
            wallPairs = ast.literal_eval(wallList)
            for wall in wallPairs:
                self.walls.append(vec(wall[0], wall[1]))
        f.close()

    def run(self):
        self.running = True
        while self.running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    if event.key == pygame.K_s: #save map
                        f = open("drawnmap.txt","w")
                        wallList = []
                        for wall in self.walls:
                            wallList.append([wall.x, wall.y])
                        f.write(str(wallList))
                        f.close()
                        print("map saved")
                    if event.key == pygame.K_l:
                        g.loadDrawn()
                        print("map loaded")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    wallpos = vec(pygame.mouse.get_pos()) // tilesize
                    if event.button == 1:
                        if wallpos in g.walls:
                            g.walls.remove(wallpos)
                        else:
                            g.walls.append(wallpos)
            pygame.display.set_caption("{:.2f}".format(g.clock.get_fps()))
            g.screen.fill(darkgray)
            g.drawGrid()
            g.draw()
            pygame.display.flip()


g = Grid(10,10)
g.run()
