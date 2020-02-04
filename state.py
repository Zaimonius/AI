import time
import random
import enum
import asyncio

class Telegram:
    def __init__(self,sender,receiver,messagetype,dispatchtime,extrainfo):
        self.sender = sender
        self.receiver = receiver
        self.messagetype = messagetype
        self.dispatchtime = dispatchtime
        self.extrainfo = extrainfo

class EntityManager:
    entityDictionary = {}
    
    def registerEntity(self,newEntity):
        self.entityDictionary[newEntity.id] = newEntity
        newEntity.setManager(self)
        print("Entity registered " + str(newEntity.id))
    @staticmethod
    def getEntity(self,ID):
        return self.entityDictionary[ID]

    def deleteEntity(self,entity):
        self.entityDictionary.pop(entity.id)

    def updateEntities(self):
        for key in self.entityDictionary:
            if self.entityDictionary[key].stateMachine.currentState != None:
                self.entityDictionary[key].Update()

class MessageDispatcher:
    priorityQ  = []
    
    def Discharge(self,receiver,msg):
        receiver.stateMachine.HandleMessage(msg)
    
    def DispatchMessage(self,senderID,receiverID,msg,delay,extraInfo,manager):
        senderEntity = manager.getEntity(manager,senderID)
        receiverEntity = manager.getEntity(manager,receiverID)
        message = Telegram(senderEntity,receiverEntity,msg,0,extraInfo)
        if delay <= 0:
            self.Discharge(receiverEntity,message)
        else:
            message.dispatchtime = time.time() + delay
            self.priorityQ.append(message)
    
    def DispatchDelayedMessages(self,manager):
        if self.priorityQ.__len__ == 0:
            while self.priorityQ[0].dispatchtime < time.time() and self.priorityQ[0].dispatchtime > 0:
                telegram = self.priorityQ[0]
                receiver = manager.getEntity(manager,telegram.receiver.id)
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
        if telegram.messagetype == MessageType.Msg_HoneyImHome:
            entity.setGlobalState(CookDinner())
        if telegram.messagetype == MessageType.Msg_StewReady:
            entity.setGlobalState(EatFood())

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
    
    def setManager(self,manager):
        self.manager = manager

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
        elif self.currentState != None:
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

#miner and wife states
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
            miner.changeState(GoSleep())
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
        

class EatFood(State):
    def Enter(self,miner):
        print("[Miner]mmm smells nice")
    
    def Execute(self,miner):
        print("[Miner]mmm delish")
        miner.setGlobalState(None)
    
    def Exit(self,miner):
        print("[Miner]thanks for the food")


class CookDinner(State):
    def Enter(self,houseWife):
        print("[Wife]making the stew!")
    
    def Execute(self,houseWife):
        print("[Wife]Stew is ready")
        houseWife.messenger.DispatchMessage(houseWife.id,"Miner",MessageType.Msg_StewReady,0,None,houseWife.manager)
        houseWife.setGlobalState(None)
    
    def Exit(self,houseWife):
        print("[Wife]back to work!")

class GoSleep(State):
    def Enter(self,miner):
        if miner.location != Location.Loc_Home:
            miner.location = Location.Loc_Home
            print("[Miner]going home")
            miner.changeState(GoSleep())
        else:
            print("Honey im home!")
            miner.messenger.DispatchMessage(miner.id,"Wife",MessageType.Msg_HoneyImHome,0,None,miner.manager)
    
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
#wife class
class HouseWife(Entity):
    # States
    #-----------------
    # HouseWork
    # - MakeBed
    # - MopFloor
    # GoToBathroom

    def __init__(self,IDValue,messenger):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(HouseWork())
        self.messenger = messenger
    
    def Update(self):
        self.stateMachine.Update()

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)

    def HandleMessage(self,message):
        return self.stateMachine.HandleMessage(message)
    
    def setGlobalState(self,newState):
        self.stateMachine.setGlobalState(newState)
#miner class
class Miner(Entity):
    # States
    #-----------------
    # DigForNuggs
    # VisitBank
    # QuenchThirst
    # GoSleep
    def __init__(self,IDValue,messenger):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(GoSleep())
        self.location = "home"
        self.goldCarried = 0
        self.moneyInBank = 0
        self.thirst = 0
        self.fatigue = 0
        self.messenger = messenger

    def Update(self):
        self.stateMachine.Update()

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)
    
    def setGlobalState(self,newState):
        self.stateMachine.setGlobalState(newState)

    def HandleMessage(self,message):
        return self.stateMachine.HandleMessage(message)





#lab classes


class Telegram:
    def __init__(self,sender,receiver,messagetype,dispatchtime,extrainfo):
        self.sender = sender
        self.receiver = receiver
        self.messagetype = messagetype
        self.dispatchtime = dispatchtime
        self.extrainfo = extrainfo

class EntityManager:
    entityDictionary = {}
    
    def registerEntity(self,newEntity):
        self.entityDictionary[newEntity.id] = newEntity
        newEntity.setManager(self)
        print("Entity registered " + str(newEntity.id))
    @staticmethod
    def getEntity(self,ID):
        return self.entityDictionary[ID]
    
    def deleteEntity(self,entity):
        self.entityDictionary.pop(entity.id)

    def updateEntities(self):
        for key in self.entityDictionary:
            if self.entityDictionary[key].stateMachine.currentState != None:
                self.entityDictionary[key].Update()

class MessageDispatcher:
    priorityQ  = []
    
    def Discharge(self,receiver,msg):
        receiver.stateMachine.HandleMessage(msg)
    
    def DispatchMessage(self,senderID,receiverID,msg,delay,extraInfo,manager):
        senderEntity = manager.getEntity(manager,senderID)
        receiverEntity = manager.getEntity(manager,receiverID)
        message = Telegram(senderEntity,receiverEntity,msg,0,extraInfo)
        if delay <= 0:
            self.Discharge(receiverEntity,message)
        else:
            message.dispatchtime = time.time() + delay
            self.priorityQ.append(message)
    
    def DispatchDelayedMessages(self,manager):
        if self.priorityQ.__len__ == 0:
            while self.priorityQ[0].dispatchtime < time.time() and self.priorityQ[0].dispatchtime > 0:
                telegram = self.priorityQ[0]
                receiver = manager.getEntity(manager,telegram.receiver.id)
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
        if telegram.messagetype == MessageType.Msg_Yes:
            entity.addFriendComing()
        if telegram.messagetype == MessageType.Msg_ImHere:
            entity.addFriendHere()
        if telegram.messagetype == MessageType.Msg_Meetup:
            telegram.extrainfo
            entity.setGlobalState()

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
    
    def setManager(self,manager):
        self.manager = manager

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
        elif self.currentState != None:
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

class MessageType(enum.Enum):
    #Message enums
    Msg_HoneyImHome = "imh"
    Msg_StewReady = "swr"
    Msg_Meetup = "meet"
    Msg_Yes = "y"
    Msg_No = "n"
    Msg_ImHere = "here"

class Location(enum.Enum):
    #Location enums
    Loc_Mine = "mine"
    Loc_Home = "home"
    Loc_Bar = "bar"
    Loc_Bank = "bank"
    Loc_Restaurant = "rest"
    Loc_Work1 = "w1"
    Loc_Work2 = "w2"
    Loc_Mall = "mall"



#state classes for persons

class Waiting(State):
    def Enter(self,person):
        print("["+str(person.id)+"] sweet sweet home")
    
    def Execute(self,person):
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(None)
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 90:
            print("["+str(person.id)+"] zzz")
            person.fatigue = 0
        elif person.thirst > 90:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 90:
            print("["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        elif person.money > 1000:
            print("["+str(person.id)+"] Shopping time!")
            person.changeState(GoShop())

    def Exit(self,person):
        print("["+str(person.id)+"] something")

class AtHome(State):
    def Enter(self,person):
        print("["+str(person.id)+"] sweet sweet home")
    
    def Execute(self,person):
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(None)
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 90:
            print("["+str(person.id)+"] zzz")
            person.fatigue = 0
        elif person.thirst > 90:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 90:
            print("["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        elif person.money > 1000:
            print("["+str(person.id)+"] Shopping time!")
            person.changeState(GoShop())
        elif 

        chore = random.randint(0,1)
        toilet = random.randint(0,10)
        if toilet == 1:
            person.changeState(GoToBathroom())
        if chore == 0:
            person.changeState(MakeBed())
        else:
            person.changeState(MopFloor())

    def Exit(self,person):
        print("["+str(person.id)+"] something")


#person class
class Person(Entity):
    
    # States
    #-----------------
    # AtHome
    # GoShop
    # 
    # Drink
    # Eat
    #-----------------
    def __init__(self,IDValue,messenger,moneyGain,fatigueGain,thirstGain,hungerGain):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(AtHome())
        self.location = Location.Loc_Home
        self.hunger = 0
        self.thirst = 0
        self.fatigue = 0
        self.money = 0
        self.messenger = messenger
        self.moneyGain = moneyGain
        self.fatigueGain = fatigueGain
        self.thirstGain = thirstGain
        self.hungerGain = hungerGain

    def Update(self):
        self.stateMachine.Update()
        hunger += hungerGain
        thirst += thirstGain
        fatigue += fatigueGain

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)
    
    def setGlobalState(self,newState):
        self.stateMachine.setGlobalState(newState)

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
    messenger.DispatchDelayedMessages(z)
    time.sleep(1)

