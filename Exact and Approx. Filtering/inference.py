import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined
import util

class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """


        if self.total() == 0.0:
            return

        tot = self.total()
        for key in self.keys():
            self[key] = float(self[key])/tot




    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """


        s_seq = []
        s_weights = []

        for item in self.items():
            s_seq.append(item[0])
            s_weights.append(float(item[1])/float(self.total()))

        x = random.random()

        for i, val in enumerate(s_seq):
            if x<=s_weights[i]:
                return val
            x-=s_weights[i]



class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)


    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).

        You can use: busters.getObservationProbability(noisyDistance, trueDistance)
        to get the probability of seeing a particular sensor reading given some
        actual distance.

        Note 1: All distances in Pac-World are Manhattan!
        Note 2: The provided method does NOT handle the jail special cases.
                You must think those through and return appropriate values!
        """

        "*** YOUR CODE HERE ***"
       
        if (ghostPosition == jailPosition and noisyDistance == None): #if were sure were in the jail
            return 1 #basically saying its def noisy because we know we are not in there
        if (ghostPosition == jailPosition or noisyDistance ==  None): # if 1 or the other is true but not both = impossible
            return 0 #were in the jail and somehow not noisy or were noisy and not in jail, contradcits each other
        else: #otherwise we get an actual probability
            #print(pacmanPosition, ghostPosition)
            distance = manhattanDistance(pacmanPosition, ghostPosition) 
            return busters.getObservationProbability(noisyDistance, distance) # return the observation prob. 
            
    


    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.
        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.  If a ghost is in jail, the sensor deterministically returns None.
        This is the only time the sensor will return None.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        location = gameState.getPacmanPosition() #get location 
        jail_location = self.getJailPosition() #get the jail location
        Possibles = self.allPositions #get the list of possible postions
        for pos in Possibles: #cycle thru the locations on the board
            curr = self.beliefs[pos] 
            self.beliefs[pos] = self.getObservationProb(observation, location, pos, jail_location) #update with new value 
            self.beliefs[pos] *= curr #multiply new times old to get the updated value

        self.beliefs.normalize() 


    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current state.
        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.

        Your agent has access to the action distribution for the ghost through
        self.getPositionDistribution.  In order to obtain the distribution over
        new positions for the ghost, given its previous position, use this line of code:

        newPosDist = self.getPositionDistribution(gameState, oldPos)

        Where oldPos refers to the previous ghost position.
        newPosDist is a DiscreteDistribution object, where for each position p
        in self.allPositions, newPosDist[p] is the probability that the ghost is
        at position p at time t + 1, given that the ghost is at position oldPos at time t

        """
        "*** YOUR CODE HERE ***" 
        jail_location = self.getJailPosition() #get the jail location 
        Possibles = self.allPositions #get the list of possible postions
        copy =  DiscreteDistribution() #make a copy to hold the working values in

        for pos in Possibles: #cycle thru the locations on the board
            if not (pos in jail_location): #we cant be in the jail 
                newPosDist = self.getPositionDistribution(gameState, pos) #new possition distribution, gives a dict
                for dist in newPosDist.items(): 
                    copy[(dist[0])] = (self.beliefs[pos] * dist[1]) + copy[dist[0]] #copy the value and add the new prob to the old 
                    

        self.beliefs = copy  
        self.beliefs.normalize()

        

    def getBeliefDistribution(self):
        return self.beliefs




class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        pos = self.legalPositions # potential positions 
        total_pos = len(pos)
        #print("HERE", total_pos)
     
        
        if total_pos != 0:
            for position in pos: #get legal positions and evenly distribute particles based on #legal positions 
                for _ in range(self.numParticles // total_pos):  #get equal values for each aprticle
                    self.particles.append((position)) 
        else:
            self.particles = []
        

        #print(self.particles)

    def observeUpdate(self, observation, gameState):
        """
        Resample particles based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.

        This method constructs a weight distribution over self.particles where the weight of a
        particle is the probability of the observation given Pacman’s position and that particle location.
        Then, we resample from this weighted distribution to construct our new list of particles.

        You should again use the function self.getObservationProb to find the probability of an observation
        given Pacman’s position, a potential ghost position, and the jail position.
        The sample method of the DiscreteDistribution class will also be useful.
        As a reminder, you can obtain Pacman’s position using gameState.getPacmanPosition(), and the jail position using self.getJailPosition()


        """
        "*** YOUR CODE HERE ***"
        jailposition = self.getJailPosition()
        pacmanPosition = gameState.getPacmanPosition()
        #distribution = self.getBeliefDistribution() 
        #print("dist", distribution)
        particles = self.particles
        #print("part", particles)
        
            
        distributed_particles  = DiscreteDistribution() # make a new distribution object for the output 
        for particle in particles: #for each particle update the output distribution with its positional probs 
            distributed_particles[particle] += self.getObservationProb(observation, pacmanPosition, particle, jailposition)  
       
    
        distributed_particles.normalize() #normalize the distribution for output purposes
        stored_distribtuion = distributed_particles #dont want to accidentally modify the distrubtion 
        
        if stored_distribtuion.total() == 0: #if there are no vlaues in the particles
            self.initializeUniformly(gameState)
        else:
            particlesOut = [] #init an empty list of output particles
            for _ in self.particles: #for every particle that exists in total list of particles
                particlesOut.append(stored_distribtuion.sample()) #get the sample value of each particle in the distribution

            self.particles = particlesOut




    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.

        As in the elapseTime method of the ExactInference class, you should use:

        newPosDist = self.getPositionDistribution(gameState, oldPos)

        This line of code obtains the distribution over new positions for the ghost, given its previous position (oldPos).
        The sample method of the DiscreteDistribution class will also be useful.


        """
        "*** YOUR CODE HERE ***"
        new_particles = [] #init holding list

        for particle in self.particles: #cycle thru all current particles 
            newPosDist = self.getPositionDistribution(gameState, particle) #get a new distribution for the particle
            new_particles.append(newPosDist.sample()) #random sample that distribution and add it to the output

        self.particles = new_particles



        #raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.

        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        output = DiscreteDistribution()
        #print("parts", self.particles)
        for state in self.particles: #just transfer the particles occurance into dictionary 
            if (state != None):
                output[state] +=1 

        output.normalize()#nomrmalize it and return  
        #print("OUTPUT", output)
        if len(output) ==0: #if there is nothing in the ouput just return an empty distribution (error otherwise)
            return DiscreteDistribution()
        else:
            return output
       


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        pos = self.legalPositions # potential positions 
        total_pos = len(pos)
        total_ghosts = self.numGhosts #get number of ghosts
        #ghost_list = []
       
        #for i in range (total_ghosts): #put all the ghosts into an iterable list for the all_combinations function use
            #ghost_list.append(i) Needed this until I discoverd repeat command and that for i in range didnt work well 
        
        if total_pos != 0:
            all_combinations = list(itertools.product(self.legalPositions, repeat=total_ghosts)) #cast result to list so we can shuffle
            random.shuffle((all_combinations))
            
            for position in all_combinations: #get legal positions and evenly distribute particles based on all combinations of positions 
                for _ in range(self.numParticles // total_pos):  #get equal values for each particle
                    self.particles.append((position)) #add it to output list
        
        else: #otherwise there are no particles
            self.particles = []
    

        #raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.

        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Resample particles based on the distance observation and Pacman's position.
        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.
        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.

        To loop over all the ghosts, use:
            for i in range(self.numGhosts):

        You can still obtain Pacman’s position using gameState.getPacmanPosition(), but to get the jail
        position for a ghost, use self.getJailPosition(i), since now there are multiple ghosts each with their own jail positions.

        As in the update method for the ParticleFilter class, you should again use the function self.getObservationProb
        to find the probability of an observation given Pacman’s position, a potential ghost position, and the jail position.
        The sample method of the DiscreteDistribution class will also be useful.

        """
        "*** YOUR CODE HERE ***"
        #jailposition = self.getJailPosition(i)
        pacmanPosition = gameState.getPacmanPosition()
        #distribution = self.getBeliefDistribution() 
        particles = self.particles
        total_ghosts = self.numGhosts
        distributed_particles  = DiscreteDistribution() # make a new distribution object for the output 

        for particle in particles: #for each particle update the output distribution with its positional probs
            first = True #init becuase first time thru we need to set 
            prob = 0 #if no ghots we can be sure the prob is 0 so init it 

            for i in range(total_ghosts):
                if first: #first time thru just want the given probability
                    prob = self.getObservationProb(observation[i], pacmanPosition, particle[i], self.getJailPosition(i))
                    first = False #reset true var
                else: #second or greater time thorugh we want to multiplpy what we have 
                    prob = prob * self.getObservationProb(observation[i], pacmanPosition, particle[i], self.getJailPosition(i))
            distributed_particles[particle] += prob  #get total prob based on each previous prob's input combined 
       
        distributed_particles.normalize() #normalize the distribution for output purposes
        stored_distribtuion = distributed_particles #dont want to accidentally modify the distrubtion (had errors here)
        if stored_distribtuion.total() == 0: #if there are no values in the particles
            self.initializeUniformly(gameState)
        else:
            particlesOut = [] #init an empty list of output particles
            for _ in self.particles: #for every particle that exists in total list of particles
                particlesOut.append(stored_distribtuion.sample()) #get the sample value of each particle in the distribution

            self.particles = particlesOut

        #self.beliefs = DiscreteDistribution()
        #raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.

        As in the last question, you can loop over the ghosts using:
            for i in range(self.numGhosts):

        Then, assuming that i refers to the index of the ghost, to obtain the distributions over new positions
        for that single ghost, given the list (prevGhostPositions) of previous positions of all of the ghosts, use:

        newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])

        """
        newParticles = []
        for oldParticle in self.particles:
            #print("OLD", oldParticle)
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            for i in range(self.numGhosts): #cycle thru for each possible ghost
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i]) #get a new distribution for the particle based on the 
                newParticle[i] = newPosDist.sample()
            #raiseNotDefined()


            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
