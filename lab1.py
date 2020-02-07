import time
import random
import enum
import asyncio
#lab classes

class Telegram:
    #telegram class that works as a struct to hold data
    def __init__(self,sender,receiver,messagetype,dispatchtime,extrainfo):
        self.sender = sender
        self.receiver = receiver
        self.messagetype = messagetype
        self.dispatchtime = dispatchtime
        self.extrainfo = extrainfo

class EntityManager:
    #contains a dictionary with every entity
    entityDictionary = {}
    #method for registering a new entity to the manager
    def registerEntity(self,newEntity):
        self.entityDictionary[newEntity.id] = newEntity
        newEntity.setManager(self)
        print("Entity registered " + str(newEntity.id))
    #getter
    @staticmethod
    def getEntity(self,ID):
        return self.entityDictionary[ID]
    #removes the entity
    def deleteEntity(self,entitityKey):
        del self.entityDictionary[entitityKey]
    #updates all the states of the entities in the dictionary
    def updateEntities(self):
        for key in list(self.entityDictionary):
            if self.entityDictionary[key].stateMachine.currentState != None:
                if self.entityDictionary[key].Dead():
                    self.deleteEntity(key)
                else:
                    self.entityDictionary[key].Update()

class MessageDispatcher:
    #contains a priority queue for messages that are supposed to be dipsatched with delay
    priorityQ  = []
    #sends the message to an entity
    def Discharge(self,receiver,msg):
        receiver.stateMachine.HandleMessage(msg)
    #converts information to a telegram
    def DispatchMessage(self,senderID,receiverID,msg,delay,extraInfo,manager):
        senderEntity = manager.getEntity(manager,senderID)
        receiverEntity = manager.getEntity(manager,receiverID)
        message = Telegram(senderEntity,receiverEntity,msg,0,extraInfo)
        if delay <= 0:
            self.Discharge(receiverEntity,message)
        else:
            message.dispatchtime = time.time() + delay
            self.priorityQ.append(message)
    #Dispatches delayed messages, called every loop
    def DispatchDelayedMessages(self):
        if len(self.priorityQ) != 0:
            if self.priorityQ[0].dispatchtime < time.time() and self.priorityQ[0].dispatchtime > 0:
                telegram = self.priorityQ[0]
                if telegram.receiver.dead == 1:
                    pass
                else:
                    self.Discharge(telegram.receiver,telegram)
                    del self.priorityQ[0]
    #method for dispatching messages to all entities in an entiitymanager
    def DispatchToAll(self,sender,msg,delay,extraInfo,manager):
        for person in list(manager.entityDictionary.values()):
            if person != sender:
                message = Telegram(sender,person,msg,0,extraInfo)
                self.Discharge(person,message)

class State:
    #state base class

    #pure virtual methods
    #makes sure that the methods are implemented
    def Enter(self):
        assert 0,"[Debug]no enter method\n"

    def Execute(self):
        assert 0,"[Debug]no execute method\n"

    def Exit(self):
        assert 0,"[Debug]no exit method\n"
    #message handling in the states
    def OnMessage(self,entity,telegram):
        if telegram.messagetype == MessageType.Msg_Yes:
            #if the person can meet they append themselves to the meeting list so that the others know th eperson is coming
            entity.meetingList.append(telegram.sender)
        if telegram.messagetype == MessageType.Msg_Meetup:
            #if an entity is invited to meetup it checks its well being and acts accordingly
            if entity.fatigue > 70 or entity.thirst > 70 or entity.hunger > 70:
                print("Text message["+str(entity.id)+"] Cant come!")
                #sends that it cant come, this doesnt do anything however
            else:
                #if it can come it sends a message back to the person who sent the meetup mesage
                print("Text message["+str(entity.id)+"] I can come!")
                entity.messenger.DispatchMessage(entity.id,telegram.sender.id,MessageType.Msg_Yes,0,telegram.extrainfo,entity.manager)
                #adds the meetinglist to the entity
                entity.meetingList = telegram.extrainfo
                #changes state
                entity.changeState(MeetUp())
        if telegram.messagetype == MessageType.Msg_CakeReady:
            #if the cake is ready go take it out of oven
            entity.changeState(TakeOutCake())

class Entity:
    #an entity contains just an id
    #not yeat implemented so that you can not have two of the same ids
    id = None
    #setter
    def setID(self,value):
        self.id = value
        print("Entity id set " + str(value))
    #constructor
    def __init__(self,idNumber):
        self.setID(idNumber)
    #id getter (not used i think)
    def ID(self):
        return self.id
    #sets the manager of an entity
    def setManager(self,manager):
        self.manager = manager
    #pure virtual
    def Update(self):
        pass
    #pure virtual
    def HandleMessage(self,telegram):
        pass

class StateMachine:
    #contains owner and current, previous and global state
    owner = None
    currentState = None
    previousState = None
    globalState = None
    #constructor for owner
    def __init__(self,ownerIn):
        self.owner = ownerIn
    #executes the current state, prioritices the global state
    def Update(self):
        if self.globalState != None:
            self.globalState.Execute(self.owner)
        elif self.currentState != None:
            self.currentState.Execute(self.owner)
    #changes the current state and the previous state
    def ChangeState(self,newState):
        assert newState, "newState is not ok"
        self.previousState = self.currentState
        self.currentState.Exit(self.owner)
        self.currentState = newState
        self.currentState.Enter(self.owner)
    #make the current state the previous state
    def revertState(self):
        self.ChangeState(self.previousState)
    #setters
    def setCurrentState(self,newState):
        self.currentState = newState

    def setGlobalState(self,newState):
        self.globalState = newState

    def setPreviousState(self,newState):
        self.previousState = newState
    #checks if the state can handle the message
    def HandleMessage(self,message):
        if self.currentState != None and self.currentState.OnMessage(self.owner,message):
            return True
        
        if self.globalState != None and self.globalState.OnMessage(self.owner,message):
            return True
        else:
            return False

class MessageType(enum.Enum):
    #Message enums
    #some of the are old(for the miner class example in book)
    Msg_HoneyImHome = "imh"
    Msg_StewReady = "swr"
    Msg_Meetup = "meet"
    Msg_Yes = "y"
    Msg_No = "n"
    Msg_ImHere = "here"
    Msg_CakeReady = "cakerdy"

class Location(enum.Enum):
    #Location enums
    #not really used in the current implementation
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
        print(str(timer) + "["+str(person.id)+"] Meeting the mates")
    #meeting with friends class
    #if they get hungry or sleepy or thirsty they go resolve their needs
    #if they dont need to go they just hang out until someone does
    def Execute(self,person):
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print(str(timer) + "["+str(person.id)+"] Dieded")
            self.Scatter(person.meetingList)
            print("[Group] Rip " + str(person.id))
        elif person.fatigue > 85:
            print(str(timer) + "["+str(person.id)+"] Gotta go, im tired")
            person.changeState(Sleep())
            self.Scatter(person.meetingList)
        elif person.thirst > 85:
            print(str(timer) + "["+str(person.id)+"] Gotta go, im thirsty")
            person.changeState(Drink())
            self.Scatter(person.meetingList)
        elif person.hunger > 85:
            print(str(timer) + "["+str(person.id)+"] Gotta go, im hungry")
            person.changeState(Eat())
            self.Scatter(person.meetingList)
        else:
            print(str(timer) + "["+str(person.id)+"] Chatting to friends")

    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] Thanks for the company")

    #If one of the entities go home they all go home and clears the meeting list
    def Scatter(self,meetingList):
        for person in list(meetingList):
            person.changeState(Sleep())
            person.meetingList = []

class Drink(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] To the bar!")
    #refills thirst and then does whatever they need after that
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] Slurp, drink and stuff")
        person.thirst = 0
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print(str(timer) + "["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print(str(timer) + "["+str(person.id)+"] Im sleepy")
            person.changeState(Sleep())
        elif person.hunger > 85:
            print(str(timer) + "["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        else:
            print(str(timer) + "["+str(person.id)+"] Gotta get back to work!")
            job = random.randint(0,2)
            if job == 1:
                person.changeState(Job1())
            elif job == 2:
                person.changeState(Job2())

    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] done dirnking - HICCUP!")

class Eat(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] sweet sweet home")
    #refills hunger and does someting else thereafter
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] Much monch very eat")
        person.hunger = 0
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print(str(timer) + "["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print(str(timer) + "["+str(person.id)+"] Im sleepy")
            person.changeState(Sleep())
        elif person.thirst > 85:
            print(str(timer) + "["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        else:
            print(str(timer) + "["+str(person.id)+"] Gotta get back to work!")
            job = random.randint(0,2)
            if job == 1:
                person.changeState(Job1())
            elif job == 2:
                person.changeState(Job2())

    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] Very good meal")

class Job1(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] Back to work!")
    #works until something else happens
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] getting the moneys at LTU")
        person.money += person.moneyGain
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print(str(timer) + "["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print(str(timer) + "["+str(person.id)+"] Im sleepy")
            person.changeState(Sleep())
        elif person.thirst > 85:
            print(str(timer) + "["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 85:
            person.changeState(Eat())
        elif person.money > 2000:
            person.changeState(GoShop())

    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] Work done for now!")

class Job2(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] Back to work!")
    #works until something else happens
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] getting the moneys at SFCS")
        person.money += person.moneyGain
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print(str(timer) + "["+str(person.id)+"] Dieded")
        elif person.fatigue > 85:
            print(str(timer) + "["+str(person.id)+"] Im sleepy")
            person.changeState(Sleep())
        elif person.thirst > 85:
            print(str(timer) + "["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        elif person.hunger > 85:
            person.changeState(Eat())
        elif person.money > 2000:
            person.changeState(GoShop())

    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] work done for now!")

class GoShop(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] shopping time!")
    #the person goes and buys a spade!
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] shoppinggg")
        person.money -= 2000
        person.items.append("spade")
        person.changeState(Sleep())

    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] done shopping")

class Bake(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] cake baking time!")
    #the person puts a cake in the oven and sends a delayed message ot himself when it is ready
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] In the oven you go!")
        person.messenger.DispatchMessage(person.id,person.id,MessageType.Msg_CakeReady,2,None,person.manager)
        person.changeState(Sleep())
    
    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] See ya later cake!")

class TakeOutCake(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] time to take cake out of owen!")
    #the person takes the cake out of the oven and achieves cake!
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] smells nice!")
        person.items.append("cake")
        person.changeState(Sleep())
    
    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] cake done!")

class Sleep(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] sweet sweet home")
    #this is the home state
    #first it replenishes all your energy and then you dec ide what you are going to do depending on multiple factors
    def Execute(self,person):
        print(str(timer) + "["+str(person.id)+"] zzz")
        person.fatigue = 0
        #if a person is too high in any stat, they die
        if person.hunger > 100 or person.thirst > 100 or person.fatigue > 100:
            person.changeState(Dead())
            print(str(timer) + "["+str(person.id)+"] Dieded")
        #if a person is thirsty it drinks
        elif person.thirst > 85:
            print(str(timer) + "["+str(person.id)+"] Im thirsty!")
            person.changeState(Drink())
        #if a person is hungry it eats
        elif person.hunger > 85:
            print(str(timer) + "["+str(person.id)+"] Im hungry!")
            person.changeState(Eat())
        #if a person is has much money it spends it on spades!!!
        elif person.money > 2000:
            print(str(timer) + "["+str(person.id)+"] Shopping time!")
            person.changeState(GoShop())
        else:
            #if not any of the above it chooses randomly wha to do
            #either it works at a job or tries to meet up with friends
            action = random.randint(0,3)
            if action == 1:
                person.changeState(Job1())
            elif action == 2:
                person.changeState(Job2())
            elif action == 0:
                person.changeState(Bake())
            else:
                #Meetup!
                if len(person.manager.entityDictionary) == 1:
                    pass
                else:
                    person.meetingList.append(person)
                    print("Text message["+str(person.id)+"] Meetup?")
                    person.messenger.DispatchToAll(person,MessageType.Msg_Meetup,0,person.meetingList,person.manager)
                    if len(person.meetingList) == 1:
                        print(str(timer) + "["+str(person.id)+"] aw mann noone wants do do anything :(")
                        person.meetingList = []
                    else:
                        person.changeState(MeetUp())
    
    def Exit(self,person):
        print(str(timer) + "["+str(person.id)+"] leaving home")

class Dead(State):
    def Enter(self,person):
        print(str(timer) + "["+str(person.id)+"] is entering heaven")
    #if a person dies he changes his dead value to 1 and the entitymanager decides to delete him
    def Execute(self,person):
        person.dead = True
        print(str(timer) + "["+str(person.id)+"] Goodbye Earth!")

#person class
class Person(Entity):
    # States
    #-----------------
    # Sleep
    # GoShop
    # MeetUp
    # Drink
    # Eat
    # Job1
    # Job2
    # Dead
    # Bake
    # TakeOutCake
    #-----------------

    #a constructor that takes values for different fields
    def __init__(self,IDValue,startState,messenger,moneyGain,fatigueGain,thirstGain,hungerGain):
        #sets the entityID
        self.setID(IDValue)
        #sets its statemachine
        self.stateMachine = StateMachine(self)
        #sets the startstate
        self.stateMachine.setCurrentState(startState)
        #locations isnt used for the moment
        self.location = Location.Loc_Home
        #stats
        self.hunger = 0
        self.thirst = 0
        self.fatigue = 0
        self.money = 0
        #the messagedispatcher to use
        self.messenger = messenger
        #stat gain fields
        self.moneyGain = moneyGain
        self.fatigueGain = fatigueGain
        self.thirstGain = thirstGain
        self.hungerGain = hungerGain
        #item list (SPADES AND CAKE BABY!)
        self.items = []
        #list field for when meeting with people
        self.meetingList = []
        #if entity is dead or alive
        self.dead = False

    #updates the state machine and gains stats
    def Update(self):
        self.stateMachine.Update()
        self.hunger += self.hungerGain
        self.thirst += self.thirstGain
        self.fatigue += self.fatigueGain

    #changes the statemachines current state
    def changeState(self,newState):
        self.stateMachine.ChangeState(newState)
    #changes the statemachines global state
    def setGlobalState(self,newState):
        self.stateMachine.setGlobalState(newState)
    #reverts the current state to the old state
    def revertState(self):
        self.stateMachine.revertState()
    #if dead, returns true
    def Dead(self):
        return self.dead
    #sends the handlemessage to the statemachine handlemessage
    def HandleMessage(self,message):
        return self.stateMachine.HandleMessage(message)

#messenger creation
messenger = MessageDispatcher()
#entities creation
x = Person("Gary",Sleep(),messenger,254,3,7,5)
y = Person("Liz",Job1(),messenger,321,5,6,3)
t = Person("Paul",Eat(),messenger,415,5,4,6)
r = Person("Mary",Drink(),messenger,213,4,3,6)
#creating the entititymanger
z = EntityManager()
#registering the entities to the manager
z.registerEntity(x)
z.registerEntity(y)
z.registerEntity(t)
z.registerEntity(r)

#simple game loop updates the entites and sends dealyed message 1 time per 2 seconds
#can be tweaked if too fast i guess, need to implement a clock for a nicer look
#if needed just remove sleep to make it super fast
timer = 7
while True:
    z.updateEntities()
    messenger.DispatchDelayedMessages()
    time.sleep(2)
    timer += 1
    if timer == 24:
        timer = 0
    print(timer)