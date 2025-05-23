# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range (self.iterations): #cycle through for the number of iterations we input
            out_vals = self.values.copy() #need copy instead of setting

            for state in self.mdp.getStates(): #updating every state 
                if not self.mdp.isTerminal(state): #unless its a termianl because those dont need to update
                    max_q_val = float("-inf") #set current max to lowest possible number
                    for action in self.mdp.getPossibleActions(state): #for all possible actions leading to new states
                        possible_q = self.computeQValueFromValues(state, action) #get the q value for the given action
                        if possible_q > max_q_val: #if its better than the current max reaplce it
                            max_q_val = possible_q
                    if max_q_val == float("-inf"): #if the max is -inf (we cant move) then were either at terminal state or weird location so return 0
                        max_q_val =0
                    out_vals[state] = max_q_val #update the output value array 
            self.values = out_vals


    



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_out = 0 #init q val
        
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action): #go through each state for next states and get the state and the prob you acatully get there
            reward_function = self.mdp.getReward(state, action, nextState) #get the reward R() function
            value = self.values[nextState] #the V() for the value 
            q_out += prob * (reward_function + (self.discount * value)) #update the current prob so that you get every input possible using the bellman eq
        return q_out

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        "*** YOUR CODE HERE ***"
        action_out = None #init output vars
        max_q_val = float("-inf")

        if (self.mdp.isTerminal(state) or (self.mdp.getPossibleActions(state)==())): #if were in terminal state there are no possible actions
            return action_out
        
        else:
            for action in self.mdp.getPossibleActions(state): #check each possible action in our state right now
                possible_q = self.computeQValueFromValues(state, action) #q value of given action
                if possible_q > max_q_val: #if it has a better q value we want to use this action
                    max_q_val = possible_q
                    action_out = action
            return action_out


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
