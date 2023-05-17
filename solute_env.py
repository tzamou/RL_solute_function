import gym
from gym import spaces
import numpy as np
import math, time
from sympy import *
import sympy
from stable_baselines3 import PPO

class Solve_Equation(gym.Env):
    def __init__(self):
        self.rewardlst = []
        self.trainreward = []
        self.maxreward_lst = []
        self.timesteps = 0
        self.testtime = 0
        self.successtime = 0
        self.totaltimestep = 0
        a = 10
        low_action = np.array([-a], dtype=np.float32)
        high_action = np.array([a], dtype=np.float32)  # z=0.027*9.8
        self.action_space = spaces.Box(low=low_action, high=high_action)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6, ), dtype=np.float32)#np.inf

        self.x = sympy.symbols('x')
    def step(self, action):
        solution = float(self.func.evalf(subs={'x': action[0]}))
        self.totaltimestep+=1

        obs = self.get_obs()
        reward = self.get_reward(solution)
        done = self.is_done(solution)

        print(f'time:{self.totaltimestep},x={action[0]} , solution={solution}, reward={reward}')
        return obs, reward, done, {}
    def reset(self):
        self.func_coef = [12,17,-1,-5,1]# ans 4,3,-1,-1
        self.func = self.get_function(func_coef=self.func_coef)
        obs = self.get_obs()
        self.data_update()
        return obs
    def render(self, mode='human', clode=False):
        pass
    def close(self):
        pass
    def get_obs(self):
        obs = np.array(self.func_coef)
        solution = self.func.evalf(subs={'x': 0})
        obs = np.append(obs,float(solution))
        return obs

    def get_reward(self,solution):
        self.rewardlst.append(-abs(solution))
        return -(abs(solution)**2)
    def is_done(self,sol):
        if round(sol,10)==0:
            return True
        else:
            return False
    def is_success(self):
        pass

    def get_function(self,func_coef):
        f = 0
        for power, coef in enumerate(func_coef, start=0):
            f += coef * self.x ** power
        return f
    def data_update(self):
        if self.testtime>0:
            print(self.rewardlst)
            print(f"[INFO] reward: {self.rewardlst[-1]}")
            print(f"[INFO] max reward: {max(self.rewardlst)}")
            print(f"[INFO] min reward: {min(self.rewardlst)}")
        self.testtime+=1
        print(f'[INFO] testtime: {self.testtime}')
        print(f'[INFO] successtime: {self.successtime}')
        print(f'--------------------------------------')

if __name__=='__main__':
    env = Solve_Equation()
    model = PPO("MlpPolicy", env, verbose=0, n_steps=3, seed=0)
    start = time.time()
    model.learn(total_timesteps=10000)
    end = time.time()
    np.save(f'reward.npy', arr=env.rewardlst)
    # while True:
    #     action=env.action_space.sample()
    #     env.step(action)



