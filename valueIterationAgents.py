# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
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
        """"*** YOUR CODE HERE ***
            all possible states are: ['TERMINAL_STATE', (0, 0), (0, 1), (0, 2)]
        """
        k = 0; #index k
        while (k < self.iterations):
            copied_values = self.values.copy()

            for state in self.mdp.getStates(): #for every state
                possible_actions = self.mdp.getPossibleActions(state) #array of possible actions
                sum_of_each_action = [] #list of sum for all actions (get max from here)
                if self.mdp.isTerminal(state) is False:
                    for action in possible_actions: #loop through every action
                        sum = 0
                        options = self.mdp.getTransitionStatesAndProbs(state, action)
                        for option in options:
                            nextState, transition = option
                            reward = self.mdp.getReward(state, action, nextState)
                            next = self.discount * self.getValue(nextState)
                            value = transition * (reward + next)
                            sum += value
                        sum_of_each_action.append(sum) #each action's sum of possible next states added
                    copied_values[state] = max(sum_of_each_action)

            self.values = copied_values
            k += 1


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
        sum = 0
        options = self.mdp.getTransitionStatesAndProbs(state, action)
        if len(options) is 0:
            return sum

        for option in options:
            nextState, transition = option
            reward = self.mdp.getReward(state, action, nextState)
            next = self.discount * self.getValue(nextState)
            value = transition * (reward + next)
            sum += value
        return sum


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        possible_actions = self.mdp.getPossibleActions(state)

        action_values = util.Counter()
        for action in possible_actions:
            action_values[action] = self.computeQValueFromValues(state, action)
        return action_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        k = 0; #index k
        index = 0;
        state_list = self.mdp.getStates()
        while (k < self.iterations):
            if index == len(state_list):
                index = 0
            state = state_list[index]
            copied_values = self.values.copy()
            possible_actions = self.mdp.getPossibleActions(state) #array of possible actions
            sum_of_each_action = [] #list of sum for all actions (get max from here)
            if self.mdp.isTerminal(state) is False:
                for action in possible_actions: #loop through every action
                    sum = 0
                    options = self.mdp.getTransitionStatesAndProbs(state, action)
                    for option in options:
                        nextState, transition = option
                        reward = self.mdp.getReward(state, action, nextState)
                        next = self.discount * self.getValue(nextState)
                        value = transition * (reward + next)
                        sum += value
                    sum_of_each_action.append(sum) #each action's sum of possible next states added
                copied_values[state] = max(sum_of_each_action)
            self.values = copied_values
            index += 1
            k += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.predecessors = {}
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        stateList = self.mdp.getStates()
        for state in stateList:
            self.predecessors[state] = self.getPredecessors(state)

        priorityQueue = util.PriorityQueue() #Initialize an empty priority queue.
        for state in stateList: #for each non-terminal state s, do:
            if self.mdp.isTerminal(state) is False:
                #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s
                diff = abs(self.values[state] - self.getMaxQValue(state))
                #Push s into the priority queue with priority -diff
                priorityQueue.update(state, -diff)

        index = 0
        for index in range(0, self.iterations):
            if priorityQueue.isEmpty():
                break
            s = priorityQueue.pop()
            if self.mdp.isTerminal(s) is False:
                candidates = []
                possible_actions = self.mdp.getPossibleActions(s)
                for action in possible_actions:
                    candidate = self.computeQValueFromValues(s, action)
                    candidates.append(candidate)
                chosen = max(candidates)
                self.values[s] = chosen #Update the value of s (if it is not a terminal state) in self.values.
            for p in self.getPredecessors(s): #For each predecessor p of s, do:
                #Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p
                diff = abs(self.values[p] - self.getMaxQValue(p))
                if diff > self.theta:
                    priorityQueue.update(p, -diff)
            index += 1

    def getMaxQValue(self, state):
        if self.mdp.isTerminal(state):
            return 0
        summations = []
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            summations.append(self.computeQValueFromValues(state, action))
        return max(summations)


    def getPredecessors(self, future):
        #may have to disregard its own state
        predecessors = set()
        stateList = self.mdp.getStates()
        for state in stateList:
            if self.mdp.isTerminal(state) is False:
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for transition in transitions:
                        nextState, prob = transition
                        if nextState == future and prob > 0:
                            predecessors.add(state)
        return predecessors
