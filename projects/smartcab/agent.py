import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import operator
from collections import namedtuple, Counter
import math

ACTIONS = [None, 'forward', 'left', 'right']

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.epsilon = 0       
        self.gamma = 0
        self.learning_rate = 1
        self.successes = []
        self.success = False
        self.q_table = {}
        self.LearningAgentState = namedtuple('LearningAgentState',['light', 'oncoming', 'left', 'next_waypoint'])
        self.i = 1        
        self.logging = {}
        self.net_reward = 0
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.logging[self.i] = (self.success, self.net_reward)
        somme = sum(self.logging[d][1] for d in self.logging)
        average = int(somme) / int(len(self.logging))
        print "Average reward for all trials = %d" %  average
        print "Successes vs Failures: %s" % Counter(self.logging[i][0] for i in self.logging).most_common()
        self.i += 1
        self.success = False
        self.net_reward = 0
        print self.logging
        
    def get_max(self, q_table, state):
        pairs = {}        
        for a in ACTIONS:
            q = self.q_table.get((state, a), 0)
            pairs[a] = q
        print pairs
        max_action = max(pairs.iteritems(), key=operator.itemgetter(1))[0]
        dictionary = {'action': max_action, 'value': pairs[max_action]}
        if pairs[max_action] == 0:
            random_a = random.choice([None, 'forward', 'left', 'right'])       
            dictionary['action'] = random_a
        print dictionary
        return dictionary
    
    def get_current_state(self, inputs, deadline, next_waypoint):
                
        state = self.LearningAgentState(light = inputs['light'], oncoming = inputs['oncoming'], left = inputs['left'], next_waypoint = next_waypoint)
        return state
    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        next_waypoint = self.next_waypoint
        self.state = self.get_current_state(inputs, deadline, next_waypoint)
        # TODO: Select action according to your policy
        choose_random_action = random.random() < self.epsilon        
        if choose_random_action:
            action = random.choice(ACTIONS)
        else:
            maximum = self.get_max(self.q_table, self.state)
            action = maximum['action']
            
        print "Recommended action: %s" % action
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.net_reward += reward
        # TODO: Learn policy based on state, action, reward
        self.learning_rate = 1/math.log(t+2)
        next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        new_state = self.get_current_state(inputs, deadline, next_waypoint)
        next_action = self.get_max(self.q_table, new_state)['action']
        self.q_table[(self.state, action)] = self.q_table.get((self.state, action), 0) + self.learning_rate * (reward + self.gamma * self.q_table.get((new_state, next_action), 0))
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]
        if location==destination:
            self.success = True
        
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
