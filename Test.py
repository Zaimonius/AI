import time

def weewoo():
    print("weewoo\n")

class State:
    def Enter(self):
        print("[Debug]enter\n")

    def Execute(self):
        print("[Debug]execute\n")

    def Exit(self):
        print("[Debug]exit\n")

class BaseGameEntity:
    #initialize with unique ID
    def __init__(self,ID):
        self.setID(ID)
    
    #setter for ID
    def setID(self,ID):
        self.id = ID
        print("[Debug]ID set\n")

    #getter for ID
    def getID(self):
        return self.id

class StateMachine(BaseGameEntity):
    def __init__(self,initState):
        self.currentState = initState
        self.gameObject = self

    def UpdateState(self):
        self.currentState.Execute(self.gameObject)

    def changeState(self,newState):
        self.currentState.Exit(self.gameObject)
        self.currentState = newState
        self.currentState.Enter(self.gameObject)

class StartState(State):
#Startstate for miner

    def __init__(self):
        return

    def Enter(self):
        print("starting game")
    
    def Execute(self,gameObject):
        print("game started")
        gameObject.changeState(DigForNuggs)


class DigForNuggs(State):
    def Enter(self,miner):
        if miner.location != "goldmine":
            print("no goldmine, aww man \n")
            miner.location = "goldmine"
    
    def Execute(self,miner):
        miner.goldCarried += 1
        print("one more nugget weoo\n")
        miner.fatigue += 5
        if miner.gold == 10:
            miner.changeState(VisitBank)
        elif miner.thirst == 100:
            miner.changeState(QuenchThirst)
        elif miner.fatigue == 100:
            miner.changeState(GoSleep)
    def Exit(self,miner):
        print("need to visit the bank now\n")

class VisitBank(State):
    def Enter(self,miner):
            if miner.location != "bank":
                print("going to bank!\n")
                miner.location = "bank"
        
    def Execute(self,miner):
        miner.moneyInBank += miner.goldCarried
        miner.goldCarried = 0
        if miner.fatigue == 100:
            miner.changeState(GoSleep)
        else:
            miner.changeState(DigForNuggs)
        
    def Exit(self,miner):
        if miner.fatigue == 100:
            print("going home to sleep\n")
        else:
            print("back to work \n")
        
class QuenchThirst(State):
    def Enter(self,miner):
        if miner.location != "saloon":
            print("gotta quench the thirst")
            miner.location = "saloon"

    def Execute(self,miner):
        miner.thirst = 0
        print("ahhhhh")
        if miner.fatigue == 100:
            miner.changeState(GoSleep)
        else:
            miner.changeState(DigForNuggs)
        
    def Exit(self,miner):
        if miner.fatigue == 100:
            print("im tired, going home")
        else:
            print("ohhhhh \n")
        
class GoSleep(State):
    def Enter(self,miner):
        if miner.location != "home":
            miner.location = "home"
            print("going home")
            miner.changeState(GoSleep)
        else:
            print("going to sleep")
    
    def Execute(self,miner):
        miner.fatigue = 0
        print("zzz")
        miner.changeState(DigForNuggs)
        
    def Exit(self,miner):
        print("Back to the mine!")


class Miner():
    stateMachine = StateMachine(StartState)
    currentState = StartState()
    location = "home"
    goldCarried = 0
    moneyInBank = 0
    thirst = 0
    fatigue = 0

    def __init__(self,ID):
        self.stateMachine.setID(ID)

    def UpdateMiner(self):
        self.thirst += 5
        self.stateMachine.UpdateState()



def errorFix():
    print("Searching for errors...")
    time.sleep(5)
    print("No errors found")

x = Miner(123)
while True:
    time.sleep(1)
    x.UpdateMiner()



weewoo()
errorFix()