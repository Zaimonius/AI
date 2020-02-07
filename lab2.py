import collections
import sys
import pygame as pg
import time
from lab2settings import * 
from lab2settings import tileSize as ts

pg.init()

class Agent(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.allSprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((ts, ts))
        self.image.fill(yellow)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        

    def move(self, dx=0, dy=0):
        self.x += dx
        self.y += dy

    def update(self):
        self.rect.x = self.x * ts
        self.rect.y = self.y * ts

class Wall(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.allSprites, game.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((ts, ts))
        self.image.fill(green)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * ts
        self.rect.y = y * ts

class Game:
    """Game class for creating a game object with pygame"""
    #game settings
    width = 1024
    height = 768
    FPS = 60
    title = "AI Pathfinding demo"
    backgroundColor = (0, 0, 0)
    tileSize = ts
    gridWidth = width/tileSize
    gridHeight = height/tileSize

    def __init__(self):
        """Constructor for the game"""
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption(self.title)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.loadMap()

    def loadMap(self):
        """method for loading a map from a text file"""
        pass

    def new(self):
        """method for making a new game setup"""
        self.allSprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.Agent = Agent(self, 10, 10)
        for x in range(10, 20):
            Wall(self, x, 5)
    
    def run(self):
        """Gameloop start method"""
        self.running = True
        while self.running:
            self.deltaTime = self.clock.tick(self.FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def draw(self):
        """method for drawing the background, grid and the sprites"""
        self.screen.fill(self.backgroundColor)
        self.drawGrid()
        self.allSprites.draw(self.screen)
        pg.display.flip()
    
    def drawGrid(self):
        """method for drawing the game grid"""
        for x in range(0, self.width, ts):
            pg.draw.line(self.screen, LightGrey, (x, 0), (x, self.height))
        for y in range(0, self.height, ts):
            pg.draw.line(self.screen, LightGrey, (0, y), (self.width, y))

    def events(self):
        """method for event handling"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN: # make this into AI controls instead!
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_LEFT:
                    self.Agent.move(dx=-1)
                if event.key == pg.K_RIGHT:
                    self.Agent.move(dx=1)
                if event.key == pg.K_UP:
                    self.Agent.move(dy=-1)
                if event.key == pg.K_DOWN:
                    self.Agent.move(dy=1)
    
    def update(self):
        """Updates all sprites in the game"""
        self.allSprites.update()

    def quit(self):
        """Quit the game and closes the window"""
        pg.quit()
        sys.exit()
    
    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass



class Graph:
    """Graph class"""
    def __init__(self):
        self.edges = {}
    
    def neighbors(self,id):
        """Returns the neighbors of the inputed id"""
        return self.edges[id]


class Queue:
    """Queue class, uses the collections library"""
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()



graph = Graph()
graph.edges = {
    'A': ['B'],
    'B': ['A', 'C', 'D'],
    'C': ['A'],
    'D': ['E', 'A'],
    'E': ['B']
}

g = Game()
g.show_start_screen()
while True:
    g.new()
    g.run()
    g.show_go_screen()