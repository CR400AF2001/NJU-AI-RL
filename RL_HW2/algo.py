import random
from collections import defaultdict

import numpy as np

from abc import abstractmethod


class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


class MyQAgent(QAgent):
    def __init__(self):
        super().__init__()
        self.learningRate = 0.5
        self.discountFactor = 0.8
        self.qTable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def select_action(self, ob):
        stateQ = self.qTable[str(ob)]
        max = []
        maxValue = stateQ[0]
        max.append(0)
        for i in range(1, 4):
            if stateQ[i] > maxValue:
                max.clear()
                maxValue = stateQ[i]
                max.append(i)
            elif stateQ[i] == maxValue:
                max.append(i)
        return random.choice(max)

    def learn(self, ob, action, reward, nextOb):
        oldQ = self.qTable[str(ob)][action]
        newQ = reward + self.discountFactor * max(self.qTable[str(nextOb)])
        self.qTable[str(ob)][action] += self.learningRate * (newQ - oldQ)
