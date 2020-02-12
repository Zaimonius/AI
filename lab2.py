import pygame
import time
from lab2settings import *
vec = pygame.math.Vector2

gridwidth = 28
gridheight = 15
width = tilesize * gridwidth
height = tilesize * gridheight

pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()


class Grid:
    def __init__(self,width,height):
        #the maps width
        self.width = width
        #the maps height
        self.height = height
        #the walls of the map
        self.walls = []
        # the directions that the agent can move in
        self.directions = [vec(1,0),vec(0,-1),vec(-1,0),vec(0,-1)]
    
    def inBounds(self,node):
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self,node):
        return node not in self.walls

    def findNeighbors(self,node):
        neibors = []
        for direction in self.directions:
            neibors.append(node + direction)
            

    def drawGrid(self):
