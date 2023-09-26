# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:27:14 2020

@author: Hang Yu
"""
#import wandb
#import schedualing as Env
#import Stacking as Env
import Q_Learning_Agent_Test as QL
import Policy_Shaping as PS
import matplotlib.pyplot as plt
import pickle
import numpy as np
import gymnasium as gym
import rlbench.gym
import panda_gym
import griddly
from griddly import gd
import RL_functions as RL
import random
from Custom_env.CustomReachEnv import CustomReachEnv1

def transform_state(state):
    return state[0:3]

def transform_action(action):
    # Switched action 3 (Left) and 4 (Right) because of the way that the user sees the robot in the environment
    if action == 1:
        envAction = [1, 0, 0]
    elif action == 2:
        envAction = [-1, 0, 0]
    elif action == 3:
        envAction = [0, -1, 0]
    elif action == 4:
        envAction = [0, 1, 0]
    elif action == 5:
        envAction = [0, 0, 1]
    elif action == 6:
        envAction = [0, 0, -1]
    else:
        envAction = [0, 0, 0]
    
    return envAction

q_learning_table_path = 'q_learning_oracle_3.pkl' 

env = CustomReachEnv1(render_mode="human",
                    render_target_position=[-0.2, 0, 0],
                    render_distance=1.0,
                    render_yaw=-230,
                    render_pitch=-50,
                    render_roll=0)

# env = gym.make('PandaReach-v3', render_mode='human')

gameName = 'PandaReach-v3'
participantID = 29

state = env.reset(seed=12)
state = transform_state(state[0]['observation'])
episodes = 100
times = 1
total_reward = [0 for i in range(episodes)]
steps = [0 for i in range(episodes)]
cnt = 0

rl = RL.RL(env.action_space.shape[0], 3)


for t in range(times):
    # print(t)
    Qagent = QL.QAgent_Test(env.action_space.shape[0], 3, gameName, participantID, epsilon = 0.2, mini_epsilon = 0.01, decay = 0.999)
    for epsd in range(episodes):
        # input(f"EPISODE: {epsd}")
        
        #GENERATE RANDOM POSITION FOR TESTING
        state_list = rl.get_state_comma(state).split(",")
        state_list = np.array([int(x) for x in state_list])
        random_position_list = np.array([random.randint(2, 17) for _ in range(len(state_list))])
        # random_position_list = [1, 13, 10]
        difference = np.subtract(random_position_list, state_list) * 3
        moves = np.array([0, 0, 0])
        max_iter = max(abs(difference))
        for i in range(max_iter):
            step1 = difference[0]
            step2 = difference[1]
            step3 = difference[2]
            
            if step1 > 0:
                moves[0] = 1
                difference[0] -= 1
            elif step1 == 0:
                moves[0] = 0
            else:
                moves[0] = -1
                difference[0] += 1
                
            if step2 > 0:
                moves[1] = 1
                difference[1] -= 1
            elif step2 == 0:
                moves[1] = 0
            else:
                moves[1] = -1
                difference[1] += 1
            
            if step3 > 0:
                moves[2] = 1
                difference[2] -= 1
            elif step3 == 0:
                moves[2] = 0
            else:
                moves[2] = -1
                difference[2] += 1
            # print(difference)
            # print(moves)
            # print("-"*50)
            env.step(moves)

            # input(f"Diff: {difference}")


        if epsd % 1000 == 0:
            print(f"Episode: {epsd}")
        start_f = cnt
        while(1):
            cnt += 1
            
            invalidState = 'invalid' in rl.get_state(state)
            if invalidState:
                state = env.reset(seed=12)
                state = transform_state(state[0]['observation'])
                total_reward[epsd] = 15000
                break

            action = Qagent.choose_action(state)
            next_state, reward, trunc, info, is_done = env.step(transform_action(action))
            steps[epsd] += 1
            next_state = transform_state(list(next_state['observation']))

            invalidNextState = 'invalid' in rl.get_state(next_state)
            if invalidNextState:
                # input(f"next_state: {epsd}, rl: {rl.get_state(next_state)}")
                state = env.reset(seed=12)
                state = transform_state(state[0]['observation'])
                total_reward[epsd] = 15000
                break

            Qagent.learning(action,reward,state,next_state)
            state = next_state
            total_reward[epsd] += reward


            if is_done['is_success'] or cnt - start_f > 1000:
                # print("ITS DONE")
                print(f"RWD: {total_reward}")
                state = env.reset(seed=12)
                state = transform_state(state[0]['observation'])
                break

Qagent.qtable.to_csv("Q_table_exp.csv")

with open(q_learning_table_path, 'wb') as pkl_file:
    pickle.dump(Qagent, pkl_file)

x=[i+1 for i in range(episodes)]
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x,np.array(total_reward)/times,color='green',label='test')
plt.savefig("Q_Learning_Oracle")

res= np.array(total_reward)/times
f = open('Q_Learning_Oracle', 'w')
for r in res:  
    f.write(str(r))  
    f.write('\n')  
f.close()