#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:07:37 2019

@author: tenzintselha
"""

import numpy as np
import random

from matplotlib import cm
from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment

import gym
from gym.spaces import Dict, Discrete, Box

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
#from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.rllib.env.multi_agent_env import MultiAgentEnv


from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog

import argparse

import os
from random import choice, randint

from colosseumrl.envs.tron.rllib import SimpleAvoidAgent
from colosseumrl.matchmaking import request_game, GameResponse
from colosseumrl.RLApp import create_rl_agent
from colosseumrl.envs.tron import TronGridClientEnvironment
from colosseumrl.envs.tron import TronGridEnvironment
from colosseumrl.rl_logging import get_logger
from copy import deepcopy


logger = get_logger()

# Our combatants are the agents we designed in the previous step
class SimpleAvoidAgent:
    def __init__(self, noise=0.1):
        self.noise = noise

    def __call__(self, env, observation):
        # With some probability, select a random action for variation
        if random.random() <= self.noise:
            return random.choice(['forward', 'right', 'left'])
        
        # Get game information
        board = observation['board']
        head = observation['heads'][0]
        direction = observation['directions'][0]
        
        # Find the head of our body
        board_size = board.shape[0]
        x, y = head % board_size, head // board_size

        # Check ahead. If it's clear, then take a step forward.
        nx, ny = env.next_cell(x, y, direction, board_size)
        if board[ny, nx] == 0:
            return 'forward'

        # Check a random direction. If it's clear, then go there.
        offset, action, backup = random.choice([(1, 'right', 'left'), (-1, 'left', 'right')])
        nx, ny = env.next_cell(x, y, (direction + offset) % 4, board_size)
        if board[ny, nx] == 0:
            return action

        # Otherwise, go the opposite direction.
        return backup
    
# A version of tron where only one agent may learn and the others are fixed
class TronRaySinglePlayerEnvironment(gym.Env):
    def __init__(self, board_size=15, num_players=4, spawn_offset=2, agent=SimpleAvoidAgent()):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None
        self.human_player = None
        self.spawn_offset = spawn_offset
        self.agent = agent

        self.renderer = TronRender(board_size, num_players, winner_player=0)
        
        self.action_space = Discrete(3)
        self.observation_space = Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })

    def reset(self):
        self.state, self.players = self.env.new_state(spawn_offset=self.spawn_offset)
        self.human_player = self.players[0]

        return self._get_observation(self.human_player)

    def _get_observation(self, player):
        return self.env.state_to_observation(self.state, player)

    def step(self, action: int):
        human_player = self.human_player

        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }

        actions = []
        for player in self.players:
            if player == human_player:
                actions.append(action_to_string[action])
            else:
                actions.append(self.agent(self.env, self._get_observation(player)))

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        observation = self._get_observation(human_player)
        reward = rewards[human_player]
        done = (human_player not in self.players) or terminal

        return observation, reward, done, {}

    def render(self, mode='human'):
        if self.state is None:
            return None

        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()
        
    def test(self, trainer, frame_time = 0.1):
        self.close()
        state = self.reset()
        done = False
        action = None
        reward = None
        cumulative_reward = 0
        
        while not done:
            # Uncomment for multiagent
            # action = trainer.compute_action(np.expand_dims(extractor.transform(obs),axis=0), prev_action=action, prev_reward=reward)
            
            # Uncomment for single agent
            action = trainer.compute_action(extractor.transform(obs), prev_action=action, prev_reward=reward)
            
            state, reward, done, results = self.step(action)
            cumulative_reward += reward
            self.render()
            
            sleep(frame_time)
        
        self.render()
        return cumulative_reward
    
# Some preprocessing to let the networks learn faster
# DUMMY
class TronExtractBoard(Preprocessor):
    def _init_shape(self, obs_space, options):
        board_size = obs_space['board'].shape[0]
        return (board_size + 4, board_size + 4, 2)
    
    def transform(self, observation):
        return observation
    
class Tron_Real_Board():
    """ Wrapper to extract just the board from the game state and simplify it for the network. """
    def __init__(self, num_players, board_size, rotate: bool = True, outside_edge: int = 2):       
        self.rotate = rotate
        self.outside_edge = outside_edge
        self.num_players = num_players
        self.board_size = board_size
 
        edge_dim = 2 * outside_edge        
        shape = (2, self.board_size + edge_dim, self.board_size + edge_dim)
 
    def transform(self, observation, rotate: int = 0):
        board = observation['board'].copy()
       
        # Make all enemies look the same
        board[board > 1] = -1
       
        # Mark where all of the player heads are
        heads = np.zeros_like(board)
       
        if (rotate != 0) and self.rotate:
            heads.ravel()[observation['heads']] += 1 + ((observation['directions'] - rotate) % 4)
           
            board = np.rot90(board, k=rotate)
            heads = np.rot90(heads, k=rotate)
           
        else:
            heads.ravel()[observation['heads']] += 1 + observation['directions']
           
        # Pad the outsides so that we know where the wall is
        board = np.pad(board, self.outside_edge, 'constant', constant_values=-1)
        heads = np.pad(heads, self.outside_edge, 'constant', constant_values=-1)
       
        # Combine together
        board = np.expand_dims(board, -1)
        heads = np.expand_dims(heads, -1)
       
        return np.concatenate([board, heads], axis=-1)
    



def tron_client(env: TronGridClientEnvironment, username: str, trainer):
    """ Our client function for the random tron client.
    Parameters
    ----------
    env : TronGridClientEnvironment
        The client environment that we will interact with for this agent.
    username : str
        Our desired username.
    """
    extractor = Tron_Real_Board(4, 13)
    # Connect to the game server and wait for the game to begin.
    # We run env.connect once we have initialized ourselves and we are ready to join the game.
    player_num = env.connect(username)
    logger.debug("Player number: {}".format(player_num))

    # Next we run env.wait_for_turn() to wait for our first real observation
    env.wait_for_turn()
    logger.info("Game started...")

    # Keep executing moves until the game is over
    terminal = False
   

    while not terminal:
        # See if there is a wall in front of us, if there is, then we will turn in a random direction.
        # number = env._player.number
        
        obs = deepcopy(env.observation)
        Mapping = { 0:"forward", 1:"right", 2:"left"}
        #print("**************************************************************************************************************************")
        #print("OBS: ", extractor.transform(obs).shape)
        # Uncomment for multiagent
        #action = Mapping[trainer.compute_action(np.expand_dims(extractor.transform(obs),axis=0))]
        
        # Uncomment for single agent
        action = Mapping[trainer.compute_action(extractor.transform(obs,rotate=player_num))]

        # We use env.step in order to execute an action and wait until it is our turn again.
        # This function will block while the action is executed and will return the next observation that belongs to us
        new_obs, reward, terminal, winners = env.step(action)
        print("Took step with action {}, got: {}".format(action, (new_obs, reward, terminal, winners)))

    # Once the game is over, we print out the results and close the agent.
    print("Player name: ", username)
    print("Player number: {}".format(player_num))
    print("Game is over. Players {} won".format(env.winners))
    logger.info("Final observation: {}".format(new_obs))

# Option
#Set to True if you want to start from a later checkpoint
def load_agent():
    
    
    # Initialize training environment
   
    ray.init()
    
    def environment_creater(params=None):
        agent = SimpleAvoidAgent(noise=0.05)
        return TronRaySinglePlayerEnvironment(board_size=13, num_players=4, agent=agent)
    
    env = environment_creater()
    tune.register_env("tron_single_player", environment_creater)
    ModelCatalog.register_custom_preprocessor("tron_prep", TronExtractBoard)
    
    # Configure Deep Q Learning with reasonable values
    config = DEFAULT_CONFIG.copy()
    config['num_workers'] = 4
    ## config['num_gpus'] = 1
    #config["timesteps_per_iteration"] = 1024
    #config['target_network_update_freq'] = 256
    #config['buffer_size'] = 100_000
    #config['schedule_max_timesteps'] = 200_000
    #config['exploration_fraction'] = 0.02
    #config['compress_observations'] = False
    #config['n_step'] = 2
    #config['seed'] = SEED
    
    #Configure for PPO
    #config["sample_batch_size"]= 100
    #config["train_batch_size"]=200
    #config["sgd_minibatch_size"]=60
    #Configure A3C with reasonable values
    
    
    
    # We will use a simple convolution network with 3 layers as our feature extractor
    config['model']['vf_share_layers'] = True
    config['model']['conv_filters'] = [(512, 5, 1), (256, 3, 2), (128, 3, 2)]
    config['model']['fcnet_hiddens'] = [256]
    config['model']['custom_preprocessor'] = 'tron_prep'
    
    # Begin training or evaluation
    #trainer = DDPGTrainer(config, "tron_single_player")
    #trainer = A3CTrainer(config, "tron_single_player")
    #trainer = DQNTrainer(config, "tron_single_player")
    trainer = PPOTrainer(config, "tron_single_player")
    
    
    trainer.restore("./ppo_checkpoint_201/checkpoint-201")
   
    return trainer#.get_policy("trainer")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", "-s", type=str, default="localhost",
                        help="Hostname of the matchmaking server.")
    parser.add_argument("--port", "-p", type=int, default=50051,
                        help="Port the matchmaking server is running on.")
    parser.add_argument("--username", "-u", type=str, default="",
                        help="Desired username to use for your connection. By default it will generate a random one.")

    logger.debug("Connecting to matchmaking server. Waiting for a game to be created.")

    args = parser.parse_args()
    trainer = load_agent()
    
    if args.username == "":
        username = "Tronado_ppo_v1"
    else:
        username = args.username

    # We use request game to connect to the matchmaking server and await a game assigment.
    game: GameResponse = request_game(args.host, args.port, username)
    logger.debug("Game has been created. Playing as {}".format(username))
    logger.debug("Current Ranking: {}".format(game.ranking))

    # Once we have been assigned a game server, we launch an RLApp agent and begin our computation
    agent = create_rl_agent(agent_fn=tron_client,
                            host=game.host,
                            port=game.port,
                            auth_key=game.token,
                            client_environment=TronGridClientEnvironment,
                            server_environment=TronGridEnvironment)
    agent(username, trainer)

    
