'''
This demo of a TAMER algorithm implmented with HIPPO Gym has been adapted
from code provided by Calarina Muslimani of the Intelligent Robot Learning Laboratory
To use this code with the default setup simply rename this file to agent.py
'''

import rlbench.gym
import gym
import time
import numpy as np
import wandb
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta, datetime
import math
import os
from time_util import ElapsedTimeThread
import Policy_Shaping as PS
import Q_Learning as QL
import RL_functions as RL


'''
This is a demo file to be replaced by the researcher as required.
This file is imported by trial.py and trial.py will call:
start()
step()
render()
reset()
close()
These functions are mandatory. This file contains minimum working versions
of these functions, adapt as required for individual research goals.
'''

def update_feedback(reward):
    if reward == "good":
        return 0.2
    elif reward == "bad":
        return -0.2
    elif reward == "none":
        return 0

epsilon_max = 1
epsilon_min = 0.1
eps_decay = 3000

weight_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(-1. * frame_idx / eps_decay)


class Agent():
    '''
    Use this class as a convenient place to store agent state.
    '''

    def start(self, env, gameName, participantID, taskName):
        '''
        Starts an OpenAI gym environment.
        Caller:
            - Trial.start()
        Inputs:
            -   game (Type: str corresponding to allowable gym environments)
        Returs:
            - env (Type: OpenAI gym Environment as returned by gym.make())
            Mandatory
        '''
        self.gameName = gameName
        self.participantID = participantID
        self.taskName = taskName

        # Get the current working directory
        cwd = os.getcwd()

        # Creating Game Directory for all runs
        self.game_path = cwd + "/userdata/" + self.gameName
        game_path_exists = os.path.exists(self.game_path)
        if not game_path_exists:
            os.makedirs(self.game_path)
            print("Game directory created: ", self.game_path)
            self.userRobotDataframe = pd.DataFrame(columns=["Participant_id", "Game_name", "Task_name", "Game_num", "Starting_position", "Timestamp", "Time_step", "Game_elapsed_time", "Task_elapsed_time", "Type_of_intervention", "Human_feedback", "State", "Action", "Reward", "Q(s,a)", "Advantage"])
            self.userRobotDataframe.to_csv((str(self.game_path)+"/"+self.gameName+"_robot_data"+".csv"), index=False)
            self.userCompletedTaskDataframe = pd.DataFrame(columns=["Participant_id", "Game_name", "Task_name", "Game_num", "Starting_position", "Game_elapsed_time", "Task_elapsed_time", "Demonstrations_used", "Feedback_used", "Total_reward", "Description"])
            self.userCompletedTaskDataframe.to_csv((str(self.game_path)+"/"+self.gameName+"_user_data"+".csv"), index=False)
            self.invalidStateActionPairsDataframe = pd.DataFrame(columns=["Participant_id", "Game_name", "Task_name", "Game_num", "Starting_position", "Time_step", "Game_elapsed_time", "Task_elapsed_time", "State", "Action", "Description"])
            self.invalidStateActionPairsDataframe.to_csv((str(self.game_path)+"/"+self.gameName+"_invalidStateActionPairs_data"+".csv"), index=False)

        if not os.path.isfile((str(self.game_path)+"/"+self.gameName+"_robot_data"+".csv")):
            self.userRobotDataframe = pd.DataFrame(columns=["Participant_id", "Game_name", "Task_name", "Game_num", "Starting_position", "Timestamp", "Time_step", "Game_elapsed_time", "Task_elapsed_time", "Type_of_intervention", "Human_feedback", "State", "Action", "Reward", "Q(s,a)", "Advantage"])
            self.userRobotDataframe.to_csv((str(self.game_path)+"/"+self.gameName+"_robot_data"+".csv"), index=False)

        if not os.path.isfile((str(self.game_path)+"/"+self.gameName+"_user_data"+".csv")):
            self.userCompletedTaskDataframe = pd.DataFrame(columns=["Participant_id", "Game_name", "Task_name", "Game_num", "Starting_position", "Game_elapsed_time", "Task_elapsed_time", "Demonstrations_used", "Feedback_used", "Total_reward", "Description"])
            self.userCompletedTaskDataframe.to_csv((str(self.game_path)+"/"+self.gameName+"_user_data"+".csv"), index=False)

        if not os.path.isfile((str(self.game_path)+"/"+self.gameName+"_invalidStateActionPairs_data"+".csv")):
            self.invalidStateActionPairsDataframe = pd.DataFrame(columns=["Participant_id", "Game_name", "Task_name", "Game_num", "Starting_position", "Time_step", "Game_elapsed_time", "Task_elapsed_time", "State", "Action", "Description"])
            self.invalidStateActionPairsDataframe.to_csv((str(self.game_path)+"/"+self.gameName+"_invalidStateActionPairs_data"+".csv"), index=False)

        self.user_experiment_path = cwd + "/userdata/" + self.gameName + "/" + self.participantID
        user_experiment_path_exists = os.path.exists(self.user_experiment_path)
        if not user_experiment_path_exists:
            os.makedirs(self.user_experiment_path)
            print("User directory created: ", self.user_experiment_path)


        self.total_reward = 0
        self.total_game_reward = 0
        self.q_sa = 0
        self.max_q_sa = 0

        self.demo = False

        # DEMO & FEEDBACK STEPS
        self.demo_steps = 100
        self.game_demo_steps = 100
        self.total_demo_steps = 100
        self.feedback_steps = 75
        self.game_feedback_steps = 75
        self.total_feedback_steps = 75

        self.human_feedback_list = [None] * 5

        # Checked in demonstration phase
        self.invalidState = False
        self.invalidNextState = False

        # Define max steps and game
        # self.max_step = 1000
        if self.taskName == 'CustomReachEnv0':
            self.max_game = 1
        else:
            self.max_game = 5
        self.end = False

        self.env = env
        
        self.action_shape = self.env.action_space.shape[0]

        if self.gameName == 'PandaPickAndPlaceDense-v3':
            self.state_space = 7
            # end-effector position (3 coordinates - xyz) + finger width (1 coordinate) + object position (3 coordinates - xyz)
            # [0:3] + [6] + [7:10]
        elif self.gameName == 'PandaPushDense-v3':
            self.state_space = 6
            # end-effector position (3 coordinates - xyz) + object position (3 coordinates - xyz)
            # [0:3] + [7:10]
        elif self.gameName == 'PandaReachDense-v3' or self.gameName == 'PandaReach-v3':
            self.state_space = 3
            # [0:3]


        self.PS = True
        if self.PS:
            np.random.seed(0)
            self.PolSh = PS.PSAgent(self.action_shape, self.state_space, self.gameName, participantID)
            self.Qagent = QL.QLAgent(self.action_shape, self.state_space, self.gameName, participantID, epsilon=0.92, mini_epsilon=0.01, decay=0.999)

        self.rl = RL.RL(self.action_shape, self.state_space)

        self.elaps_time = ElapsedTimeThread()
       
        wandb.init(project='FeedbackLearning',name=self.participantID+" "+self.gameName)

        #wandb variables
        self.human_feeback_bad_total = 0
        self.human_feeback_good_total = 0
        self.human_feeback_bad = 0
        self.human_feeback_bad_discrete = False
        self.human_feeback_good_discrete = False
        self.human_feeback_good = 0
        self.human_feeback_total = 0
        
        self.game_num=1
        self.game_num_list=[]

        self.data_HF_bad_list = []
        self.data_HF_good_list = []

        self.data_HF_bad_total_list = []
        self.data_HF_good_total_list = []
        
        self.feedback_reward_cumulative = 0

        self.timestep_learn=0
        self.learning_steps = 0
        self.time_step_list=[]

        self.total_test_bar = []

        self.demo_list=[]
        self.total_democount=0
        self.demo_bool = False
        self.demo_reward_cumulative = 0

        self.oldDataframe = pd.DataFrame()
        self.userRobotDataframe = pd.DataFrame()
        self.userCompletedTaskDataframe = pd.DataFrame()
        self.invalidStateActionPairsDataframe = pd.DataFrame()

        self.exp_start_time = timer()
        self.game_start_time = timer()

        self.game_elapsed_time_list=[]
        
        self.startingPoint = True
        self.startingObservation = None
        self.done = False
        self.seedNumber = 1
        print("start complete")

        print("Initiating game: ", self.gameName)
        print("Experiments starting for: ", self.participantID)
        print("Task Name: ", self.taskName)
        return

    def step(self, human_action, human_feedback):
        '''
        Takes a game step.
        Caller:
            - Trial.take_step()
        Inputs:
            - env (Type: OpenAI gym Environment)
            - action (Type: int corresponding to action in env.action_space)
        Returns:
            - envState (Type: dict containing all information to be recorded for future use)
              change contents of dict as desired, but return must be type dict.
        '''

        self.human_feeback_bad_discrete = False
        self.human_feeback_good_discrete = False
        self.demo_bool = False
        self.time_step += 1
        self.trial_timer = timedelta(seconds=timer()-self.exp_start_time).seconds

        if self.demo == False:
            if self.time_step == 1:
                self.state = list(self.first_state[0]['observation'])
                self.last_state = np.copy(self.state)
                self.last_state = self.transform_state(list(self.last_state))
                time.sleep(1.5)
            else:
                self.last_state = np.copy(self.state)

            self.invalidState = 'invalid' in self.rl.get_state(self.last_state)

            if self.invalidState:
                self.invalidStateActionPairsDataframe.append(
                    {   "Participant_id": self.participantID,
                        "Game_name": self.gameName,
                        "Task_name": self.taskName,
                        "Game_num": self.game_num,
                        "Starting_position": self.startingObservation,
                        "Time_step": self.time_step,
                        "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                        "Task_elapsed_time": self.trial_timer,
                        "State": self.rl.get_state_comma(self.last_state) if self.time_step != 1 else self.startingObservation,
                        "Action": action,
                        "Description": "Invalid State",
                    },
                    ignore_index=True
                ).to_csv(str(self.game_path)+"/"+self.gameName+"_invalidStateActionPairs_data"+".csv", mode='a', header=False, index=False)
                self.env.reset(seed=self.seedNumber)
                self.startingPoint = True
                self.time_step = 0
                return

            self.cnt += 1
            weight = weight_by_frame(self.cnt)
            
            prob = self.Qagent.action_prob(self.last_state) + weight * np.asarray(self.PolSh.action_prob(self.last_state))

            prob = np.asarray(prob)
            action = np.random.choice(np.flatnonzero(prob == prob.max())) + 1
            envAction = self.transform_action(action)

            feedback = update_feedback(human_feedback)
            self.feedback_reward_cumulative += feedback

            next_state, reward, trunc, info, done = self.env.step(envAction)

            next_state = list(next_state['observation'])
            next_state = self.transform_state(next_state)

            self.invalidNextState = 'invalid' in self.rl.get_state(next_state)
            if self.invalidNextState:
                self.invalidStateActionPairsDataframe.append(
                    {   "Participant_id": self.participantID,
                        "Game_name": self.gameName,
                        "Task_name": self.taskName,
                        "Game_num": self.game_num,
                        "Starting_position": self.startingObservation,
                        "Time_step": self.time_step,
                        "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                        "Task_elapsed_time": self.trial_timer,
                        "State": self.rl.get_state_comma(self.last_state) if self.time_step != 1 else self.startingObservation,
                        "Action": action,
                        "Description": "Invalid State",
                    },
                    ignore_index=True
                ).to_csv(str(self.game_path)+"/"+self.gameName+"_invalidStateActionPairs_data"+".csv", mode='a', header=False, index=False)
                self.env.reset(seed=self.seedNumber)
                self.startingPoint = True
                self.time_step = 0
                return

            self.human_feedback_list.pop(0)
            self.human_feedback_list.append({'state': self.last_state, 'action': action})

            if action != 0 and self.feedback_steps > 0:
                for value in self.human_feedback_list:
                    if value != None:
                        self.PolSh.learning(value['action'], feedback, value['state'])

                self.q_sa, self.max_q_sa = self.Qagent.learning(action, reward, self.last_state, next_state)
                self.timestep_learn+=1
                self.total_game_reward += reward

            self.state = next_state
            self.total_reward += reward

        else:
            if self.demo_steps == 0:
                self.demo = False

            feedback_demo = 1
            self.demo_reward_cumulative += 1
            if self.time_step == 1:
                self.state = list(self.first_state[0]['observation'])
                self.last_state = np.copy(self.state)
                self.last_state = self.transform_state(list(self.last_state))
                time.sleep(1.5)
            else:
                self.last_state = np.copy(self.state)

            envAction = self.transform_action(human_action)

            next_state, reward, _, info, done = self.env.step(envAction)

            next_state = list(next_state['observation'])
            next_state = self.transform_state(next_state)

            self.invalidState = 'invalid' in self.rl.get_state(self.last_state)
            self.invalidNextState = 'invalid' in self.rl.get_state(next_state)

            if self.invalidState or self.invalidNextState:
                self.invalidStateActionPairsDataframe.append(
                    {   "Participant_id": self.participantID,
                        "Game_name": self.gameName,
                        "Task_name": self.taskName,
                        "Game_num": self.game_num,
                        "Starting_position": self.startingObservation,
                        "Time_step": self.time_step,
                        "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                        "Task_elapsed_time": self.trial_timer,
                        "State": self.rl.get_state_comma(self.last_state) if self.time_step != 1 else self.startingObservation,
                        "Action": human_action,
                        "Description": "Invalid State",
                    },
                    ignore_index=True
                ).to_csv(str(self.game_path)+"/"+self.gameName+"_invalidStateActionPairs_data"+".csv", mode='a', header=False, index=False)
                self.env.reset(seed=self.seedNumber)
                self.startingPoint = True
                self.time_step = 0
                return

            if human_action != 0 and self.demo_steps > 0:
                # print(f"State: {self.rl.get_state_comma(self.last_state)}")
                self.PolSh.learning(human_action, feedback_demo, self.last_state)
                self.q_sa, self.max_q_sa = self.Qagent.learning(human_action, reward, self.last_state, next_state)

                self.demo_steps -= 1
                self.game_demo_steps -= 1
                self.total_democount+=1
                self.demo_bool = True
                self.total_game_reward += reward

            action = human_action
            self.state = next_state
            self.total_reward += reward

        if done['is_success']:
            self.game_num_list.append(self.game_num)
            self.game_elapsed_time_list.append(timedelta(seconds=timer()-self.game_start_time).seconds)
            self.human_feeback_bad=0
            self.human_feeback_good=0
            self.done = True
            self.startingPoint = True


        self.Qagent.qtable.to_csv(str(self.user_experiment_path)+"/"+self.gameName+"_"+self.taskName+"_Qtable"+".csv")
        self.PolSh.feedback.to_csv(str(self.user_experiment_path)+"/"+self.gameName+"_"+self.taskName+"_PStable"+".csv")

        self.time_step_list.append(self.time_step)
        self.data_HF_bad_list.append(self.human_feeback_bad)
        self.data_HF_good_list.append(self.human_feeback_good)
        self.data_HF_bad_total_list.append(self.human_feeback_bad_total)
        self.data_HF_good_total_list.append(self.human_feeback_good_total)


        if human_feedback=='good':
            self.human_feeback_good+=1
            self.human_feeback_good_total+=1
            self.human_feeback_total +=1
            self.human_feeback_good_discrete = True
            self.feedback_steps -= 1
            self.game_feedback_steps -= 1
        if human_feedback=='bad':
            self.human_feeback_bad+=1
            self.human_feeback_bad_total+=1
            self.human_feeback_total+=1
            self.human_feeback_bad_discrete = True
            self.feedback_steps -= 1
            self.game_feedback_steps -= 1
        
        # DATA THAT WERE SAVED IN THE PREVIOUS PROJECT
        self.oldDataframe = self.oldDataframe.append(
            {   "Participant_id": self.participantID,
                "Game_name": self.gameName,
                "time_step": self.time_step,
                "Game_num": self.game_num,
                "Human_action": human_action,
                "Human_feedback": human_feedback,
                "Total_bad_feeback": self.human_feeback_bad_total,
                "Total_good_feeedback": self.human_feeback_good_total,
                "Total_feedback": self.human_feeback_total,
                "Total_Reward": self.total_reward,
                "Total_demo_steps_left": self.demo_steps,
                "Total_feedback_steps_left": self.feedback_steps,
                "elapsed_time":timedelta(seconds=timer()-self.exp_start_time),
                "bad_feedback_bool": self.human_feeback_bad_discrete,
                "good_feedback_bool":self.human_feeback_good_discrete,
                "demo_bool": self.demo_bool
            },
            ignore_index=True
        )

        self.oldDataframe.to_csv(str(self.user_experiment_path)+"/"+str(self.participantID)+"_"+self.gameName+"_old_data"+".csv")

        # USER DATA
        if human_feedback != 'none' or human_action != 0:
            if human_feedback != 'none':
                for value in self.human_feedback_list:
                    if value != None:
                        self.userRobotDataframe.append(
                            {   "Participant_id": self.participantID,
                                "Game_name": self.gameName,
                                "Task_name": self.taskName,
                                "Game_num": self.game_num,
                                "Starting_position": self.startingObservation,
                                "Timestamp": datetime.now(),
                                "Time_step": self.time_step,
                                "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                                "Task_elapsed_time": self.trial_timer,
                                "Type_of_intervention": 'feedback',
                                "Human_feedback": human_feedback,
                                "State": self.rl.get_state_comma(value['state']) if self.time_step != 1 else self.startingObservation,
                                "Action": value['action'],
                                "Reward": reward,
                                "Q(s,a)": self.q_sa,
                                "Advantage": float(self.max_q_sa - self.q_sa),
                            },
                            ignore_index=True
                        ).to_csv(str(self.game_path)+"/"+self.gameName+"_robot_data"+".csv", mode='a', header=False, index=False)
            elif human_action != 0:
                self.userRobotDataframe.append(
                    {   "Participant_id": self.participantID,
                        "Game_name": self.gameName,
                        "Task_name": self.taskName,
                        "Game_num": self.game_num,
                        "Starting_position": self.startingObservation,
                        "Timestamp": datetime.now(),
                        "Time_step": self.time_step,
                        "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                        "Task_elapsed_time": self.trial_timer,
                        "Type_of_intervention": 'demonstration',
                        "Human_feedback": human_feedback,
                        "State": self.rl.get_state_comma(self.last_state) if self.time_step != 1 else self.startingObservation,
                        "Action": human_action,
                        "Reward": reward,
                        "Q(s,a)": self.q_sa,
                        "Advantage": float(self.max_q_sa - self.q_sa),
                    },
                    ignore_index=True
                ).to_csv(str(self.game_path)+"/"+self.gameName+"_robot_data"+".csv", mode='a', header=False, index=False)
        else:
            if action != 0:
                self.userRobotDataframe.append(
                    {   "Participant_id": self.participantID,
                        "Game_name": self.gameName,
                        "Task_name": self.taskName,
                        "Game_num": self.game_num,
                        "Starting_position": self.startingObservation,
                        "Timestamp": datetime.now(),
                        "Time_step": self.time_step,
                        "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                        "Task_elapsed_time": self.trial_timer,
                        "Type_of_intervention": 'None',
                        "Human_feedback": human_feedback,
                        "State": self.rl.get_state_comma(self.last_state) if self.time_step != 1 else self.startingObservation,
                        "Action": action,
                        "Reward": reward,
                        "Q(s,a)": self.q_sa,
                        "Advantage": float(self.max_q_sa - self.q_sa),
                    },
                    ignore_index=True
                ).to_csv(str(self.game_path)+"/"+self.gameName+"_robot_data"+".csv", mode='a', header=False, index=False)

        wandb.log(
            {
            "Total_rewards":self.total_reward,
            "human_feeback_total":self.human_feeback_total,
            "human_feeback_good_ingame":self.human_feeback_good,
            "human_feeback_bad_ingame":self.human_feeback_bad,
            "human_feeback_bad_total":self.human_feeback_bad_total,
            "human_feeback_good_total":self.human_feeback_good_total,
            "Game_num": self.game_num,
            "Demo_count": self.total_democount,
            "elapsed_time_seconds":timedelta(seconds=timer()-self.exp_start_time).seconds,
            "elapsed_time_minutes":(timedelta(seconds=timer()-self.exp_start_time).seconds)/60,
            "Reward_from_demo":self.demo_reward_cumulative,
            "Reward_feedback_cumulative": self.feedback_reward_cumulative,
            }
            )
        
        if done['is_success'] or self.trial_timer >= 180:
            self.userCompletedTaskDataframe.append(
                {   "Participant_id": self.participantID,
                    "Game_name": self.gameName,
                    "Task_name": self.taskName,
                    "Game_num": self.game_num,
                    "Starting_position": self.startingObservation,
                    "Game_elapsed_time": timedelta(seconds=timer()-self.game_start_time).seconds if self.time_step != 1 else 0,
                    "Task_elapsed_time": self.trial_timer,
                    "Demonstrations_used": int(self.total_demo_steps - self.game_demo_steps),
                    "Feedback_used": int(self.total_feedback_steps - self.game_feedback_steps),
                    "Total_reward": self.total_game_reward,
                    "Description": "Done" if done['is_success'] else "Time"
                },
                ignore_index=True
            ).to_csv(str(self.game_path)+"/"+self.gameName+"_user_data"+".csv", mode='a', header=False, index=False)
            self.game_start_time = timer()
            self.game_num+=1
            # DEMO & FEEDBACK STEPS
            self.game_demo_steps = 100
            self.game_feedback_steps = 75
            self.total_game_reward = 0


        envState = {'observation': next_state, 'reward': reward, 'done': done['is_success'], 'info': info, 'agentAction': action}
        return envState

    def render(self):
        '''
        Gets render from gym.
        Caller:
            - Trial.get_render()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            - return from env.render('rgb_array') (Type: npArray)
              must return the unchanged rgb_array
        '''
        return self.env.render()

    def reset(self):
        '''
        Resets the environment to start new episode.
        Caller:
            - Trial.reset()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        if self.game_num > self.max_game:
            self.end = True

        if self.done:
            self.done = False

        if self.PS:
            self.cnt = 0
            self.time_step = 0
            self.first_state = self.env.reset(seed=self.seedNumber)
            self.observation = self.first_state
            self.startingObservation = self.rl.get_state_comma(self.transform_state(self.observation[0]['observation']))
            self.human_feedback_list = [None] * 5
        else:
            self.elaps_time.stop()
            self.elaps_time.join
            self.observation = self.env.reset(seed=self.seedNumber)
            self.startingObservation = self.rl.get_state_comma(self.transform_state(self.observation[0]['observation']))
            self.human_feedback_list = [None] * 5

    def close(self):
        '''
        Closes the environment at the end of the trial.
        Caller:
            - Trial.close()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        self.env.close()

    def take_training_step(self, action):
        observation = self.env.step(action)
        observation = observation[0]['observation']
        self.startingObservation = self.rl.get_state_comma(self.transform_state(observation))

    def transform_action(self, action):
        # Switched action 3 (Left) and 4 (Right) because of the way that the user sees the robot in the environment
        if self.action_shape == 3:
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
        elif self.action_shape == 4:
            if action == 1:
                envAction = [1, 0, 0, 0]
            elif action == 2:
                envAction = [-1, 0, 0, 0]
            elif action == 3:
                envAction = [0, -1, 0, 0]
            elif action == 4:
                envAction = [0, 1, 0, 0]
            elif action == 5:
                envAction = [0, 0, 1, 0]
            elif action == 6:
                envAction = [0, 0, -1, 0]
            elif action == 7:
                envAction = [0, 0, 0, 1]
            elif action == 8:
                envAction = [0, 0, 0, -1]
            else:
                envAction = [0, 0, 0, 0]

        return envAction
    
    def transform_state(self, state):
        if self.state_space == 3:
            state = state[0:3]
        elif self.state_space == 6:
            state = np.concatenate((state[0:3], state[7:10]), axis=0)
        elif self.state_space == 7:
            state = np.concatenate((state[0:3], [state[6]], state[7:10]), axis=0)
        return state


if __name__ == '__main__':
    pass