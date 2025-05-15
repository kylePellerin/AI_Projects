# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter() #init self.values as an empty dictionary style data structure 

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if ((state, action) not in self.values): #if we havent seen this return zero
            return 0.0
        else:
            return self.values.get((state, action)) #otherwise return 


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): #if temrinal state were done
            return 0.0
        else: 
          max_q = float("-inf")
          for action in self.getLegalActions(state): #go through all actions and find the best q value in there
            action_Q = self.getQValue(state, action) 
            if action_Q > max_q: #rest the storage varaibles 
                max_q = action_Q
          return max_q
              

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.

          break ties randomly using random.choice()
          0 may be the best choice if all other paths are 0
        """
        "*** YOUR CODE HERE ***"
        if not self.getLegalActions(state): #if there are no actions return none
            return None
        else:
          max_best_action = float("-inf")
          best_actions = [] #init an array in case we have multiple options
          for action in self.getLegalActions(state): # for each possible action get the q value of it
              action_Q = self.getQValue(state, action) 
              if action_Q == max_best_action: #if actions have the same q value add them to the list!
                  best_actions.append(action)
              if action_Q > max_best_action: #if a new action is better than currnet best clear the list and add the action
                  best_actions.clear() #clear out the list because theres a new best
                  best_actions =[action]
                  max_best_action = action_Q
          if len(best_actions) ==1: #if we only have one action return it
              return best_actions[0]
          else: #if there was a tie make a random choice 
              return random.choice(best_actions)
                  
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action 
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions: #if theres no action return the default none
            return action
        else:
            if util.flipCoin(self.epsilon): #flip a coin and if it lands on prob epislon choose a random move
                return random.choice(legalActions)
            else: #otherwise return the best action based on q values at the current state
                return self.computeActionFromQValues(state)
            
        #util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        current_q = self.getQValue(state, action) #current Q(s, a)
        next_q = self.computeValueFromQValues(nextState) #future Q(s', a')
        q_func = current_q + (self.alpha * (reward + (self.discount * next_q) - current_q)) #q learning equation 
        self.values[(state, action)] = q_func #set the state action tuples value to the q func

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
