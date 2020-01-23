import time

class State:
    def Enter(self):
        assert 0,"[Debug]no enter method\n"

    def Execute(self):
        assert 0,"[Debug]no execute method\n"

    def Exit(self):
        assert 0,"[Debug]no exit method\n"

class Entity:
    id = 0
    nextValidID = None

    def setID(self,value):
        self.id = value
        print("Entity id set")

    def __init__(self,idNumber):
        self.setID(idNumber)

    def ID(self):
        return self.id
    
    def Update(self):
        pass
      
class StateMachine:
    owner = None
    currentState = None
    previousState = None
    globalState = None

    def __init__(self,ownerIn):
        self.owner = ownerIn

    def Update(self):
        if self.globalState != None:
            self.globalState.Execute(self.owner)
        
        if self.currentState != None:
            self.currentState.Execute(self.owner)

    def ChangeState(self,newState):
        assert newState, "newState is not ok"
        self.previousState = self.currentState
        self.currentState.Exit(self.owner)
        self.currentState = newState
        self.currentState.Enter(self.owner)

    def revertState(self):
        self.ChangeState(self.previousState)
    
    def setCurrentState(self,newState):
        self.currentState = newState
    
    
    def setGlobalState(self,newState):
        self.globalState = newState
    
    
    def setPreviousState(self,newState):
        self.previousState = newState
    
class DigForNuggs(State):
    def Enter(self,miner):
        if miner.location != "goldmine":
            print("no goldmine, aww man \n")
            miner.location = "goldmine"
    
    def Execute(self,miner):
        miner.goldCarried += 1
        print("one more nugget weoo\n")
        miner.fatigue += 2
        miner.thirst += 5
        if miner.goldCarried == 10:
            miner.changeState(VisitBank())
            print("gotta go bank")
        elif miner.thirst == 10:
            miner.changeState(QuenchThirst())
            print("i need me slurp")
        elif miner.fatigue == 10:
            miner.changeState(GoSleep())
            print("im tired!")
    def Exit(self,miner):
        print("leaving the digsite\n")

class VisitBank(State):
    def Enter(self,miner):
            if miner.location != "bank":
                print("going to bank!\n")
                miner.location = "bank"
        
    def Execute(self,miner):
        miner.moneyInBank += miner.goldCarried
        miner.goldCarried = 0
        if miner.fatigue == 10:
            miner.changeState(GoSleep())
            print("im tired!")
        else:
            miner.changeState(DigForNuggs())
            print("back to work")
        
    def Exit(self,miner):
        if miner.fatigue == 10:
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
        if miner.fatigue == 10:
            miner.changeState(GoSleep())
        else:
            miner.changeState(DigForNuggs())
        
    def Exit(self,miner):
        if miner.fatigue == 10:
            print("im tired, going home")
        else:
            print("ohhhhh \n")
        
class GoSleep(State):
    def Enter(self,miner):
        if miner.location != "home":
            miner.location = "home"
            print("going home")
        else:
            print("going to sleep")
    
    def Execute(self,miner):
        miner.fatigue = 0
        print("zzz")
        miner.changeState(DigForNuggs())
        
    def Exit(self,miner):
        print("Back to the mine!")

class Miner(Entity):
    def __init__(self,IDValue):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(GoSleep())
        self.location = "home"
        self.goldCarried = 0
        self.moneyInBank = 0
        self.thirst = 0
        self.fatigue = 0

    def Update(self):
        self.thirst += 1
        self.stateMachine.Update()

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)


x = Miner(123)

while True:
    x.Update()
    time.sleep(3)