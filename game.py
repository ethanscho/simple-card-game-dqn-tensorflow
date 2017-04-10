import time
import random
import numpy as np
import tensorflow as tf 

#from state import State
from dqn import DQN
from ddqn import DDQN
from simple_game_state import CardGameState

MAX_EPISODES = 3000000.0
GAME_HISTORY_SIZE = 2000.0

class GameManager():
    def __init__(self):
        # Init game state
        self.episode = 0.0
        self.win_counter = 0.0

        self.state = CardGameState(self)
        self.brain = DDQN()

        self.episode_reward = 0
        self.game_history = list()

    def update(self, dt):
        pass

    def auto_play(self):
        while self.episode < MAX_EPISODES:
            action = self.brain.get_action(self.state)

            action_to_store = np.zeros(3)
            action_to_store[action] = 1

            self.state.process(action)
            # receive game result
            reward = self.state.reward
            done = self.state.terminal

            self.episode_reward += reward

            self.brain.train(self.state, self.state.s_t, action_to_store, reward, self.state.s_t1, done)
            
            self.state.t += 1

            self.state.update()

            if done:
                self.episode += 1
                win_rate = 0.0

                if self.episode_reward == 1:
                    self.game_history.append(1)
                else:
                    self.game_history.append(0)

                if len(self.game_history) < GAME_HISTORY_SIZE:
                    win_rate = np.sum(self.game_history) / float(len(self.game_history)) * 100.0
                else:
                    self.game_history.pop(0)
                    win_rate = np.sum(self.game_history) / GAME_HISTORY_SIZE * 100.0

                print("Episode {} | Win Rate = {}".format(self.episode, win_rate))

                self.brain.write_summary(win_rate, self.episode)

                self.episode_reward = 0
                self.state.reset()

game_manager = GameManager()
game_manager.auto_play()