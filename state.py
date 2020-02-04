import time
import random
import enum
import asyncio
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
    
    def deleteEntity(self,entitityKey):
        del self.entityDictionary[entitityKey]

    def updateEntities(self):
        for key in list(self.entityDictionary):
            if self.entityDictionary[key].stateMachine.currentState != None:
                if self.entityDictionary[key].Dead():
                    self.deleteEntity(key)
                else:
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
            telegram.senderEntity.addFriendComing()
        if telegram.messagetype == MessageType.Msg_No:
            telegram.senderEntity.addFriendNotComing()
        if telegram.messagetype == MessageType.Msg_ImHere:
            telegram.senderEntity.addFriendHere()
        if telegram.messagetype == MessageType.Msg_Meetup:
            if entity.fatigue > 70 or entity.thirst > 70 or entity.hunger > 70:
                print("Text message["+str(entity.id)+"] Cant come!")
                entity.messenger.DispatchMessage(entity.id,telegram.senderEntity.id,MessageType.Msg_No,0,0,entity.messenger)
            else:
                print("Text message["+str(entity.id)+"] I can come!")
                entity.messenger.DispatchMessage(entity.id,telegram.senderEntity.id,MessageType.Msg_Yes,0,0,entity.messenger)
                entity.host = telegram.senderEntity
                entity.changeState(MeetUp())

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
class MeetUp(State):
    def Enter(self,person):
        print("["+str(person.id)+"] Meeting the mates")
    
    def Execute(self,person):
        person.messenger.DispatchMessage(person.id,person.host.id,MessageType.Msg_ImHere,0,0,person.messenger)
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print("["+str(person.id)+"] Gotta go, im tired")
        elif person.thirst > 85:
            print("["+str(person.id)+"] Gotta go, im thirsty")
            person.changeState(Drink())
        elif person.hunger > 85:
            print("["+str(person.id)+"] Gotta go, im hungry")
            person.changeState(Eat())
        else:
            print("["+str(person.id)+"]Chatting to friends")
            person.changeState(AtHome())
    
    def Exit(self,person):
        print("["+str(person.id)+"] Thanks for the company")

class Drink(State):
    def Enter(self,person):
        print("["+str(person.id)+"] To the bar!")
    
    def Execute(self,person):
        print("["+str(person.id)+"] Slurp, drink and stuff")
        person.thirst = 0
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print("["+str(person.id)+"] Im sleepy")
            person.changeState(AtHome())
        elif person.hunger > 85:
            print("["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        else:
            print("["+str(person.id)+"] Gotta get back to work!")
            person.revertState()

    def Exit(self,person):
        print("["+str(person.id)+"] done dirnking")

class Eat(State):
    def Enter(self,person):
        print("["+str(person.id)+"] sweet sweet home")
    
    def Execute(self,person):
        print("["+str(person.id)+"] Much monch very eat")
        person.hunger = 0
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print("["+str(person.id)+"] Im sleepy")
            person.changeState(AtHome())
        elif person.thirst > 85:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())

    def Exit(self,person):
        print("["+str(person.id)+"] Very good meal")

class Job1(State):
    def Enter(self,person):
        print("["+str(person.id)+"] Back to work!")

    def Execute(self,person):
        print("["+str(person.id)+"]getting the moneys at LTU")
        person.money += person.moneyGain
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print("["+str(person.id)+"] Im sleepy")
            person.changeState(AtHome())
        elif person.thirst > 85:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 85:
            person.changeState(Eat())
        elif person.money > 2000:
            person.changeState(GoShop())

    def Exit(self,person):
        print("["+str(person.id)+"] Work done for now!")

class Job2(State):
    def Enter(self,person):
        print("["+str(person.id)+"] Back to work!")

    def Execute(self,person):
        print("["+str(person.id)+"]getting the moneys at SFCS")
        person.money += person.moneyGain
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print("["+str(person.id)+"] Im sleepy")
            person.changeState(AtHome())
        elif person.thirst > 85:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 85:
            person.changeState(Eat())
        elif person.money > 2000:
            person.changeState(GoShop())

    def Exit(self,person):
        print("["+str(person.id)+"] work done for now!")

class GoShop(State):
    def Enter(self,person):
        print("["+str(person.id)+"] shopping time!")

    def Execute(self,person):
        print("["+str(person.id)+"] shoppinggg")
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print("["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print("["+str(person.id)+"] zzz")
            person.fatigue = 0
        elif person.thirst > 85:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 85:
            print("["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        else:
            person.money -= 2000
            person.items.append("spade")
            person.changeState(AtHome())
        

    def Exit(self,person):
        print("["+str(person.id)+"] something")

class AtHome(State):
    def Enter(self,person):
        print("["+str(person.id)+"] sweet sweet home")
    
    def Execute(self,person):
        print("["+str(person.id)+"] zzz")
        person.fatigue = 0
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print("["+str(person.id)+"] Dieded")
        elif person.thirst > 85:
            print("["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 85:
            print("["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        elif person.money > 2000:
            print("["+str(person.id)+"] Shopping time!")
            person.changeState(GoShop())
        else:
            job = random.randint(0,2)
            if job == 1:
                person.changeState(Job1())
            elif job == 2:
                person.changeState(Job2())
            else:
                #meetup
                pass

    def Exit(self,person):
        print("["+str(person.id)+"] leaving home")

class Dead(State):
    def Enter(self,person):
        print("["+str(person.id)+"] is entering heaven")
    
    def Execute(self,person):
        person.dead = 1
        print("["+str(person.id)+"]Goodbye Earth!")

#person class
class Person(Entity):
    # States
    #-----------------
    # AtHome
    # GoShop
    # MeetUp
    # Drink
    # Eat
    # Job1
    # Job2
    # Dead
    #-----------------

    def __init__(self,IDValue,startState,messenger,moneyGain,fatigueGain,thirstGain,hungerGain):
        self.setID(IDValue)
        self.stateMachine = StateMachine(self)
        self.stateMachine.setCurrentState(startState)
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
        self.friendsHere = 0
        self.friendsComing = 0
        self.host = None
        self.items = []
        self.dead = 0

    def Update(self):
        self.stateMachine.Update()
        self.hunger += self.hungerGain
        self.thirst += self.thirstGain
        self.fatigue += self.fatigueGain

    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)
    
    def setGlobalState(self,newState):
        self.stateMachine.setGlobalState(newState)
    
    def revertState(self):
        self.stateMachine.revertState()

    def Dead(self):
        if self.dead == 1:
            return True
        else:
            return False

    def HandleMessage(self,message):
        return self.stateMachine.HandleMessage(message)
    
    def addFriendComing(self):
        self.friendsComing += 1
    
    def addFriendNotComing(self):
        self.friendsComing -= 1
    
    def addFriendHere(self):
        self.friendsHere += 1
    
    def friendLeft(self):
        self.friendsHere -= 1


messenger = MessageDispatcher()
x = Person("Gary",AtHome(),messenger,254,9,7,8)
y = Person("Liz",Job1(),messenger,321,8,7,9)
z = EntityManager()

z.registerEntity(x)
z.registerEntity(y)

#simple game loop updates 1 time per second
while True:
    z.updateEntities()
    messenger.DispatchDelayedMessages(z)
    time.sleep(1)

