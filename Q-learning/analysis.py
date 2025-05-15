# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.005 #make the noise really low so the chance of falling in the pits has less of a factor 
    return answerDiscount, answerNoise

def question3a():
    answerDiscount = 0.9 #high discount means reward prioritized
    answerNoise = 0.01 #low noise so cliffs not a problem
    answerLivingReward = -5 #dont take many steps!
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE' 
   

def question3b():
    answerDiscount = .3 #low discount to encourage exploration of the further path 
    answerNoise = .3 #moderate noise to help avoid cliff
    answerLivingReward = -.1 #get there kinda quick but not extremly fast
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    answerDiscount = .95 #high discount to encouarge going for bigger reward
    answerNoise = .01 #low to encourage going by cliff
    answerLivingReward = -.5 #so we dont take the long way around the board
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    answerDiscount = .95 #high discount to encouarge going for bigger reward
    answerNoise = .3 #moderate nosie to discourage cliff
    answerLivingReward = -.1 #dont need to get there too fast but cant dilly dally 
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
    

def question3e():
    answerDiscount = -10 #never go to the answer
    answerNoise = 0 #know exactly where youre going
    answerLivingReward = 11 #living is better than any reward
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
