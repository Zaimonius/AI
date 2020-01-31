import time
import random
import enum
import asyncio


class MessageType(enum.Enum):
    Msg_HoneyImHome = "imh"
    Msg_StewReady = "swr"

class Telegram:
    def __init__(self,sender,receiverID,messagetype,dispatchtime,extrainfo):
        self.sender = sender
        self.receiverID = receiverID
        self.messagetype = messagetype
        self.dispatchtime = dispatchtime
        self.extrainfo = extrainfo

class EntityManager:
    entityDictionary = {}

    def registerEntity(self,newEntity):
        self.entityDictionary[str(newEntity.id)] = newEntity
        print("Entity registered " + str(newEntity.id))
    
    def getEntity(self,ID):
        return self.entityDictionary[str(ID)]
    
    def deleteEntity(self,entity):
        self.entityDictionary.pop(str(entity.id))

    def updateEntities(self):
        for key in self.entityDictionary:
            self.entityDictionary[key].Update()

class MessageDispatcher:
    priorityQ  = []
    
    def Discharge(self,receiver,msg):
        receiver.HandleMessage(msg) #TODO is this right?
    
    def DispatchMessage(self,sender,receiverID,msg,delay,extraInfo):
        message = Telegram(sender,receiverID,msg,0,extraInfo)
        receiverEntity = EntityManager.getEntity(EntityManager,receiverID)

        if delay < 0:
            self.Discharge(receiverEntity,message)
        else:
            message.dispatchtime = time.time + delay
            self.priorityQ.append(message)
    
    def DispatchDelayedMessages(self):
        while self.priorityQ[0].dispatchtime < time.time and self.priorityQ[0].dispatchtime > 0:
            telegram = self.priorityQ[0]
            receiver = EntityManager.getEntity(EntityManager,telegram.ID)
            self.Discharge(receiver,telegram)
            del self.priorityQ[0]

class State:
    def Enter(self):
        assert 0,"[Debug]no enter method\n"

    def Execute(self):
        assert 0,"[Debug]no execute method\n"

    def Exit(self):
        assert 0,"[Debug]no exit method\n"

    def OnMessage(self,entity,telegram):
        pass

class Entity:
    id = 0
    nextValidID = None

    def setID(self,value):
        self.id = value
        print("Entity id set " + str(value))

    def __init__(self,idNumber):
        self.setID(idNumber)

    def ID(self):
        return self.id
    
    def Update(self):
        pass

    def HandleMessage(self,telegram):
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

    def HandleMessage(self,message):
        if self.currentState != None and self.currentState.OnMessage(self.owner,message):
            return True
        
        if self.globalState != None and self.globalState.OnMessage(self.owner,message):
            return True
        else:
            return False

class WaitForFood(State):
    def Enter(self,miner):
        print("Honey im home!")
        miner.messenger.DispatchMessage(0,miner,"Wife",MessageType.Msg_HoneyImHome,0,None)
    
    def Execute(self,miner):
        pass


        
    
        
class DigForNuggs(State):
    def Enter(self,miner):
        if miner.location != "goldmine":
            print("[Miner]no goldmine, aww man")
            miner.location = "goldmine"
    
    def Execute(self,miner):
        miner.goldCarried += 2
        print("[Miner]one more nugget weoo")
        miner.fatigue += 15
        miner.thirst += 25
        if miner.goldCarried > 10:
            miner.changeState(VisitBank())
            print("[Miner]gotta go bank")
        elif miner.thirst > 100:
            miner.changeState(QuenchThirst())
            print("[Miner]i need me slurp")
        elif miner.fatigue > 100:
            miner.changeState(WaitForFood())
            print("[Miner]im tired!")
    def Exit(self,miner):
        print("[Miner]leaving the digsite")

class VisitBank(State):
    def Enter(self,miner):
            if miner.location != "bank":
                print("[Miner]going to bank!")
                miner.location = "bank"
        
    def Execute(self,miner):
        miner.moneyInBank += miner.goldCarried
        miner.goldCarried = 0
        if miner.fatigue == 100:
            miner.changeState(GoSleep())
            print("[Miner]im tired!")
        else:
            miner.changeState(DigForNuggs())
            print("[Miner]back to work")
        
    def Exit(self,miner):
        print("[Miner]leaving bank naow")
        
class QuenchThirst(State):
    def Enter(self,miner):
        if miner.location != "saloon":
            print("[Miner]gotta quench the thirst")
            miner.location = "saloon"

    def Execute(self,miner):
        miner.thirst = 0
        print("[Miner]ahhhhh")
        if miner.fatigue == 10:
            miner.changeState(GoSleep())
        else:
            miner.changeState(DigForNuggs())
        
    def Exit(self,miner):
        print("[Miner]Whisky sure is nice")
        
class GoSleep(State):
    def Enter(self,miner):
        if miner.location != "home":
            miner.location = "home"
            print("[Miner]going home")
        else:
            print("[Miner]going to sleep")
    
    def Execute(self,miner):
        miner.fatigue = 0
        print("[Miner]zzz")
        if miner.thirst > 10:
            miner.changeState(QuenchThirst())
        else:
            miner.changeState(DigForNuggs())

    def Exit(self,miner):
        print("[Miner]Back to the mine!")

class MakeBed(State):
    def Enter(self,houseWife):
        print("[Wife]time to make the bed!")
    
    def Execute(self,houseWife):
        print("[Wife]making the bed...")
        houseWife.changeState(HouseWork())
        
    def Exit(self,houseWife):
        print("[Wife]bed made!")

class MopFloor(State):
    def Enter(self,houseWife):
        print("[Wife]time to mop the floor!")
    
    def Execute(self,houseWife):
        print("[Wife]mopping the floor...")
        houseWife.changeState(HouseWork())

        
    def Exit(self,houseWife):
        print("[Wife]floor mopped!")
    
class GoToBathroom(State):
    def Enter(self,houseWife):
        print("[Wife]need to use the ladies room")
    
    def Execute(self,houseWife):
        print("[Wife]taking a huge DUMP...")
        houseWife.changeState(HouseWork())
        
    def Exit(self,houseWife):
        print("[Wife]dump complete!")

class HouseWork(State):
    def Enter(self,houseWife):
        print("[Wife]time to the next chore!")
    
    def Execute(self,houseWife):
        chore = random.randint(0,1)
        toilet = random.randint(0,10)
        if toilet == 1:
            houseWife.changeState(GoToBathroom())
        if chore == 0:
            houseWife.changeState(MakeBed())
        else:
            houseWife.changeState(MopFloor())
        
    def Exit(self,houseWife):
        print("[Wife]Phew!")

class HouseWife(Entity):
    # States
    #-----------------
    # HouseWork
    # - MakeBed
    # - MopFloor
    # GoToBathroom

    def __init__(self,IDValue,Messenger):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(HouseWork())
        self.Messenger = Messenger
    
    def Update(self):
        self.stateMachine.Update()

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)

    def HandleMessage(self,message):
        return self.stateMachine.HandleMessage(message)

class Miner(Entity):
    # States
    #-----------------
    # DigForNuggs
    # VisitBank
    # QuenchThirst
    # GoSleep
    def __init__(self,IDValue,Messenger):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(GoSleep())
        self.location = "home"
        self.goldCarried = 0
        self.moneyInBank = 0
        self.thirst = 0
        self.fatigue = 0
        self.Messenger = Messenger

    def Update(self):
        self.stateMachine.Update()

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)

    def HandleMessage(self,message):
        return self.stateMachine.HandleMessage(message)

messenger = MessageDispatcher()
x = Miner("Miner",messenger)
y = HouseWife("Wife",messenger)
z = EntityManager()
z.registerEntity(x)
z.registerEntity(y)

#simple game loop updates 1 time per second
while True:
    z.updateEntities()
    time.sleep(1)

