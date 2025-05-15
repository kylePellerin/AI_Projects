# search.py
#Kyle Pellerin
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

from inspect import stack
from operator import add
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    #initalize the variables 
    root = problem.getStartState()
    fringe = util.Stack()
    visited= set()
    path =[]
    
    #put succesors of root into the stack 
    for item in problem.getSuccessors(root):
        fringe.push(([item[1]],(item))) 

    while not fringe.isEmpty():
        path, node = (fringe.pop()) 
        if problem.isGoalState(node[0]):
            return path #should return the list of actions
        if node[0] not in visited:
            #explores new nodes and pushes them to the stack 
            visited.add(node[0])
            for item in problem.getSuccessors(node[0]):
                if item[0] not in visited:
                    fringe.push(((path + [item[1]]),item))
        
    return path




def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    #initalize the variables 
    root = problem.getStartState()
    fringe = util.Queue() #changed from DFS 
    visited= set()
    path =[]
    
    #put succesors of root into the stack 
    for item in problem.getSuccessors(root): 
        fringe.push(([item[1]],(item))) 

    visited.add(root)

    while not fringe.isEmpty():
        path, node = (fringe.pop()) 
        if problem.isGoalState(node[0]):
            return path #should return the list of actions
        if node[0] not in visited:
            #explores new nodes and pushes them to the stack 
            visited.add(node[0])
            for item in problem.getSuccessors(node[0]):
                if item[0] not in visited: #if node not visited 
                    fringe.push(((path + [item[1]]),item))
        
    return path


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    fringe = util.PriorityQueue() #changed from DFS and BFS
    visited= set()
    path =[]
    
    #puts root in the stack
    
    fringe.push((0, root, []),0)  #took new appraoch here as advised by LA's


    while not fringe.isEmpty():
        cost, state, path = (fringe.pop()) 
        if problem.isGoalState(state):
            return path #should return the list of actions
        if state not in visited:
            #explores new nodes and pushes them to the stack 
            visited.add(state)
            for new_state, action, add_cost in problem.getSuccessors(state):
                if new_state not in visited: #adds the new state to the stack 
                    tot_cost = cost+add_cost
                    tot_path = path + [action]
                    fringe.push((tot_cost,new_state,tot_path), tot_cost)
        
    return path



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    root = problem.getStartState()
    fringe = util.PriorityQueue() #changed from DFS and BFS
    visited= set()
    path =[]
    
    #puts root in the stack
    
    fringe.push((0, root, []),0)  #took new appraoch here as advised by LA's


    while not fringe.isEmpty():
        cost, state, path = (fringe.pop()) 
        if problem.isGoalState(state):
            return path #should return the list of actions
        if state not in visited:
            #explores new nodes and pushes them to the stack 
            visited.add(state)
            for new_state, action, add_cost in problem.getSuccessors(state):
                if new_state not in visited: #adds the new state to the stack 
                    tot_cost = cost+add_cost
                    tot_path = path + [action]
                    fringe.push((tot_cost,new_state,tot_path), tot_cost+ heuristic(new_state,problem))
                    #line above updated to add heurisitic value into cost function 
    return path
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
