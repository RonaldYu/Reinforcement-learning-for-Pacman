# mlLearningAgents.py
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py
from __future__ import division
from pacman import Directions
from game import Agent
import random
import game
import util

import copy
import operator
# QLearnAgent
#

# A* algorithm for making features
# Calculate the distance between two point
# This code was written by Cheng-Yuan Yu, me


def hv_distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


# Following this class, we create the object(node) for using a* algorithm
class Node:

    def __init__(self, current_point):
        # node's parent node
        self.parent = None
        # node's coordinate
        self.node = current_point
        # g is the cost of the path from the start node to here
        self.g = 0
        # h is a heuristic that estimates the cost of the cheapest path from
        # here to the goal.
        self.h = 0
        # f = g + h
        self.f = 0
        # node's children's node
        self.children = []

    # method to calculate h cost of the node
    def calculcate_h(self, end_point):
        self.h = hv_distance(self.node, end_point)

    # method to calculate g cost of the node
    def calculate_g(self):
        self.g = self.parent.g + hv_distance(self.node, self.parent.node)

    # method to calculate f cost of the node
    def calculate_f(self):
        self.f = self.h + self.g


# execute a* algorithm
# Input:
#     current_node: the object Node
#     end_point: tuple, (x, y), the coordinate of the destination
#     obstacles: the list of coordinates (x, y), which is not allowed to pass
#     open_set: the set of Node's objects, which are candidate we are going to find deeply
#     closed_set: the set of Node's objects, which are selected from open_set
# Output:
#     the set of Node's objects which we will get optimal path from
def A_Star(current_node, end_point, obstacles, open_set, closed_set):
    # extract the current coordinate from the object Node
    current_point = current_node.node
    # computing the neighbors of the current
    candidates = list(set([(current_point[0] + i, current_point[1] + j) for i in [-1, 0, 1]
                           for j in [-1, 0, 1] if abs(i) + abs(j) == 1]) - set(obstacles))
    # find the boundary of the map
    boundary = [(min(i), max(i)) for i in zip(*obstacles)]
    # filter some candidates which are not in the map
    candidates = [i for i in candidates if i[0] > boundary[0][0] and i[0]
                  < boundary[0][1] and i[1] > boundary[1][0] and i[1] < boundary[1][1]]

    # Don't need to find the point which has been in the close list
    candidates = set(candidates) - set([i.node for i in closed_set])

    # If it meet the destination or there is no other point in the open set,
    # it stop finding deeply.
    # if current_point != end_point and len(open_set) > 0:
    if len(open_set) > 0:
        try:
            # looking for each candidate
            for element in candidates:
                # make it the object of Node
                candidate_node = Node(element)
                # set its parent
                candidate_node.parent = current_node
                # calculate its h
                candidate_node.calculcate_h(end_point)
                # calculate its g
                candidate_node.calculate_g()
                # calculate its f
                candidate_node.calculate_f()

                # if a node with the same position as candidate is in the open set which has a lower f than candidate, skip this candidate
                # if a node with the same position as candidate is in the
                # closed set which has a lower f than successor, skip this
                # candidate
                criterion = [i for i in open_set.union(closed_set) if (
                    i.f < candidate_node.f) and (i.node == candidate_node.node)]
                # Otherwise, add this candidate to th open set and to the
                # children node of the current node
                if len(criterion) == 0:
                    open_set.add(candidate_node)
                    current_node.children.append(candidate_node)

            # remove the current node from the open set
            open_set.remove(current_node)
            # add the current node into the closed set
            closed_set.add(current_node)
            # if arriving end_point, return closed_set
            if current_node.node == end_point:
                return closed_set
            # find the node in the open set with the minimum f
            next_node = min(open_set, key=lambda i: i.f)
            return A_Star(next_node, end_point, obstacles, open_set, closed_set)
        # If there is some error, it will return the closed set
        except:
            return closed_set
    # If it meet the destination or there is no point in the open set, it
    # return the closed set.
    else:
        return closed_set


# Find the path from the result of A_Star/the closed set.
# Input:
#    end_point: tuple, (x, y), which is the coordinate of destination.
#    closed_set: the set of Node's objects
#    selective_path: the list of tuples which are coordinates
#        ex. if our destination is (3, 3)
#        the output is = [(1, 1), (1, 2), (1, 3), (2,3)]
# we will move from (1, 1) to (3, 3) according to (1, 1) -> (1, 2) -> (1,
# 3) -> (2,3)
def backforward(end_node, closed_set, selective_path):

    # looking for all nodes in the closed_set
    for i in closed_set:
        # if once finding end_point is a child of the node, then this node will
        # be put in the front of selective_path.
        if end_node in i.children:
            selective_path.insert(0, i)

            # look backforward for the child of this node.
            return backforward(i, closed_set, selective_path)
    # if finding the beginning, it will return selective_path
    return selective_path


class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.001, epsilon=0.1, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # these weights which I am going to train
        self.weights = [0, 0, 0]
        # map of width
        self.map_width = 0
        # save the state of the last step
        self.memory_s = {'my_position': None, 'walls_position': None, 'food_position': None,
                         'ghosts_position': None, 'legal_action': None, 'action': None}
        # record how many times of updating
        self.n = 0

        # is it the first movement in a game ?
        self.is_first_move = True

        # these weights for reward
        self.weights_reward = [-1, 5]
    # Accessor functions for the variable episodesSoFars controlling learning

    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self, value):

        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    # update alpha during training
    def update_Alpha(self, value):

        if value > 0 and self.alpha > 0:
            self.alpha = 1 / (100 + value)

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # this function is to get positions (a list of tuples)
    def get_position(self, bits):
        bits = list(str(bits))
        x_i = 0
        y_i = 0
        position = []
        for i in bits:
            if i == 'T':
                position.append((x_i, y_i))
            x_i += 1
            if i != 'T' and i != 'F':
                x_i = 0
                y_i += 1

        return [(i, self.map_width - 1 - j) for i, j in position]

    # calculate how many walls surrounding the location
    def num_walls_one_step(self, location, walls_position):
        x, y = location
        list_location = [(i, j) for i in [-1, 0, 1]
                         for j in [-1, 0, 1] if abs(i) + abs(j) == 1]
        list_location = [(x + i, y + j) for i, j in list_location]
        list_location = [i for i in list_location if i not in walls_position]
        return len(list_location)

    # to calculate the value for the second and third feature
    def shortest_distance(self, get_mylocation, walls_position, target_position, for_food):
        target_length = {}

        # using absolute distance to find 4 targets with shorter distance from the pacman location
        # avoiding redundant computation for A* algorithm
        list_target = [
            (ii, hv_distance(ii, get_mylocation)) for ii in target_position if ii not in walls_position]

        list_target = sorted(
            list_target, key=operator.itemgetter(1), reverse=False)
        list_target = list_target[:min(4, len(list_target))]
        list_target = [i for i, j in list_target]

        if len(list_target) > 0:
            for i_target in list_target:
                # set current location as the object of Node
                current_node = Node(get_mylocation)
                # initialise the open set for A* algorithm
                open_set = set([current_node])
                # initialise the closed set for A* algorithm
                closed_set = set([])
                # execute A* algorithm to find the closed set
                closed_set = A_Star(current_node, i_target,
                                    walls_position, open_set, closed_set)

                # reconstruct the closed set to be the optimal route for the
                # next target and store this route to the list selective_path
                try:
                    target_node = random.choice(
                        [i for i in closed_set if i.node == i_target])

                    selective_path = [i.node for i in backforward(
                        target_node, closed_set, [])]
                    target_length[i_target] = len(selective_path)
                except:
                    target_length[i_target] = 10

            if for_food:
                # to get the feature value for the second features
                target_length = [(i, self.num_walls_one_step(
                    i, walls_position) * j)for i, j in target_length.items()]
                target_length = sorted(
                    target_length, key=operator.itemgetter(1), reverse=False)
                return target_length[0][1]
            else:
                return float(min(target_length.values()))

        else:
            return 10.0
    # what's the position if the pacman make the movement at the location,
    # my_position

    def find_position(self, my_position, move):
        x, y = my_position
        if move == 'East':
            return (x + 1, y)
        elif move == 'West':
            return (x - 1, y)
        elif move == 'North':
            return (x, y + 1)
        elif move == 'South':
            return (x, y - 1)
        else:
            return (x, y)
    # Q-learning
    # get features for pacman given pacman's state

    def get_features(self, my_position, walls_position, food_position, ghosts_position):

        dist_closest_ghost = self.shortest_distance(
            my_position, walls_position, ghosts_position, False)

        dist_closest_food = self.shortest_distance(
            my_position, walls_position, food_position, True)

        return [1.0, dist_closest_food, min(dist_closest_ghost, 4) - 3.5]

    # given feature, calculate q value
    def get_QValue(self, s, a):

        # return Q(state, action)
        # if the pacman make the action, a, in the location, s['my_position'],
        next_position = self.find_position(s['my_position'], a)
        # given features and action, get features
        features = self.get_features(
            next_position, s['walls_position'], s['food_position'], s['ghosts_position'])

        # calculate the q value
        q_value = 0.0
        for i in range(0, len(features)):
            q_value += features[i] * self.weights[i]
        return q_value
    # given the state (location), get maximum Q value by selecting the legal
    # action

    def get_max_QValue(self, s):

        list_q_values = []

        for a in s['legal_action']:

            list_q_values.append((a, self.get_QValue(s, a)))

        return sorted(list_q_values, key=operator.itemgetter(1), reverse=True)[0]

    # given the state (location), get the action which has the maximum q value from the list of legal actions
    # it sometimes select an action randomly (exploration)
    def get_action(self, s):
        # generate random variable from uniform(0,1)
        uniform_rv = random.uniform(0, 1)
        # whether it choose an action randomly
        if self.epsilon > uniform_rv and self.alpha != 0:
            return random.choice(s['legal_action'])
        else:
            output = self.get_max_QValue(s)
            return output[0]
    # update weights

    def update(self, s, a, next_s):
        # update learning rate
        self.update_Alpha(self.n)

        # get q value for the last step
        current_QValue = self.get_QValue(s, a)
        # get features for the last step
        features = self.get_features(
            s['my_position'], s['walls_position'], s['food_position'], s['ghosts_position'])

        if features[2] != 0.5:
            # if the last step, the pacman was close to ghosts
            # make more learning rate for the third weight
            adjusted_learning_rate = 10
        else:
            # if not, use the original learning rate for the third weight
            adjusted_learning_rate = 1

        # calculate the reward for the last step
        reward = self.weights_reward[0] * features[1] + \
            self.weights_reward[1] * features[2]

        # calculate the max q value for the current step
        max_QValue = self.get_max_QValue(next_s)

        # calculate modification
        dif_w = reward + self.gamma * max_QValue[1] - current_QValue

        # update three of weights
        for i in range(0, len(self.weights)):

            if i == 2:
                self.weights[i] += adjusted_learning_rate * \
                    self.alpha * dif_w * features[i]
            else:
                self.weights[i] += self.alpha * dif_w * features[i]

    # record the state
    def record_s(self, state):

        s = {}
        s['my_position'] = state.getPacmanPosition()
        s['walls_position'] = self.get_position(state.getWalls())
        s['food_position'] = self.get_position(state.getFood())
        s['ghosts_position'] = [(int(i), int(j))
                                for i, j in state.getGhostPositions()]
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        s['legal_action'] = legal

        return s

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move

    def getAction(self, state):

        # get the legal actions except STOP
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # store the current state
        next_s = self.record_s(state)

        # if this is the first step
        # just randomly select the action

        if self.is_first_move:

            # if this step is in the first game
            # just randomly select action
            if self.n == 0:
                self.map_width = state.getWalls().height
                pick = random.choice(legal)

            # if this is the first step in a game except the first game
            # we use the get_action function and do not update the weight in
            # this step.
            else:
                pick = self.get_action(next_s)

            # set is_first_move = False, let the following steps in this game
            # update the weights
            self.is_first_move = False

        else:

            # if this step is not the first step in a game, we will update now.
            # after the first step, we start training our weights for each step
            self.update(self.memory_s, self.memory_s['action'], next_s)

            # get the action for the current
            pick = self.get_action(next_s)

        # save the current state for the next step
        self.memory_s.update(next_s)
        # save the action the pacman will make in the current step
        self.memory_s['action'] = pick

        # record the times of updating
        self.n += 1

        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):

        # for the first step of the next game
        # because we are not going to update weights in the first step of each
        # game
        self.is_first_move = True
        

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)
