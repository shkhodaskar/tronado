import numpy as np
import random
#import pdb

from matplotlib import cm
from time import sleep
from colosseumrl.envs.tron import TronGridEnvironment, TronRender
from colosseumrl.envs.tron.rllib import TronRaySinglePlayerEnvironment

import gym
from gym.spaces import Dict, Discrete, Box

import ray
from ray import tune
#from ray.rllib.agents.pg import PGTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
#from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.models import ModelCatalog

SEED = 12345
#np.random.seed(SEED)

# Our combatants are the agents we designed in the previous step
class SimpleAvoidAgent:
    def __init__(self, noise=0.1):
        self.noise = noise
        #self.trainer = None
# = DQNTrainer(DEFAULT_CONFIG, environment)
        #self.trainer.restore("./dqn_model_v2/checkpoint_6700/checkpoint-6700")

    def __call__(self, env, observation, environment):
        print(observation)
        # With some probability, select a random action for variation
        #print(environment)
        #self.trainer = DQNTrainer(DEFAULT_CONFIG, environment)
        #pdb.set_trace()
        #self.trainer.restore("./dqn_model_v2/checkpoint_6700/checkpoint-6700")
        #action = self.trainer.compute_action(observation)
        #print("********************************************************************************************************************************")
        #print("ACTION",action)
        #return action
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

class SmartAgent:
    def __init__(self,smart_agent):
        self.smart_agent = smart_agent

    def __call__(self, env, observation):
        return self.smart_agent.compute_action(observation)

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
                actions.append(self.agent(self.env, self._get_observation(player), "tron_single_player"))

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
        
    def test(self, trainer, frame_time = 0.05):
        self.close()
        state = self.reset()
        done = False
        action = None
        reward = None
        cumulative_reward = 0
        
        while not done:
            action = trainer.compute_action(state, prev_action=action, prev_reward=reward)
            
            state, reward, done, results = self.step(action)
            #print(state)
            cumulative_reward += reward
            self.render()
            
            sleep(frame_time)
        
        self.render()
        return cumulative_reward

# Some preprocessing to let the network learn faster
class TronExtractBoard(Preprocessor):
    def _init_shape(self, obs_space, options):
        board_size = env.observation_space['board'].shape[0]
        return (board_size + 4, board_size + 4, 2)

    def transform(self, observation):
        board = observation['board']
        
        # Make all enemies look the same
        board[board > 1] = -1
        
        # Mark where all of the player heads are
        heads = np.zeros_like(board)
        heads.ravel()[observation['heads']] += 1 + observation['directions']
        
        # Pad the outsides so that we know where the wall is
        board = np.pad(board, 2, 'constant', constant_values=-1)
        heads = np.pad(heads, 2, 'constant', constant_values=-1)
        
        # Combine together
        board = np.expand_dims(board, -1)
        heads = np.expand_dims(heads, -1)
        
        return np.concatenate([board, heads], axis=-1)

# Option
LOAD_FROM_CHECKPOINT = False

# Initialize training environment
ray.shutdown()
ray.init()

def environment_creater(params=None):
    agent = SimpleAvoidAgent(noise=0.05)
    #agent = DQNTrainer(DEFAULT_CONFIG,"tron_single_player")
    #agent.load("./dqn_model_v2/checkpoint_6700/checkpoint-6700")
    return TronRaySinglePlayerEnvironment(board_size=13, num_players=4, agent=agent)

env = environment_creater()
print("***************************************************************************************************************************************************************************")
tune.register_env("tron_single_player", environment_creater)

ModelCatalog.register_custom_preprocessor("tron_prep", TronExtractBoard)

# Configure Deep Q Learning with reasonable values
config = DEFAULT_CONFIG.copy()
#config['num_workers'] = 4
# config['num_gpus'] = 1
#config["timesteps_per_iteration"] = 1024
#config['lambda'] = .7
#config['target_network_update_freq'] = 256
#config['buffer_size'] = 100_000
#config['schedule_max_timesteps'] = 200_000
#config['exploration_fraction'] = 0.4
#config['compress_observations'] = False
#config['n_step'] = 3
#config['seed'] = SEED

# We will use a simple convolution network with 3 layers as our feature extractor
config['model']['vf_share_layers'] = True
config['model']['conv_filters'] = [(512, 5, 1), (256, 3, 2), (128, 3, 2)]
config['model']['fcnet_hiddens'] = [256]
config['model']['custom_preprocessor'] = 'tron_prep'

# Begin training or evaluation
trainer = PPOTrainer(config, "tron_single_player")
num_epoch = 10000
test_epoch = 2

if LOAD_FROM_CHECKPOINT:
#    np.random.seed(42)
    trainer.restore("./ppo_model/checkpoint_400/checkpoint-400")
    for epoch in range(num_epoch):
        print("Training iteration: {}".format(epoch), end='')
        res = trainer.train()
        print(f", Average reward: {res['episode_reward_mean']}")

        if epoch % test_epoch == 0:
            reward = env.test(trainer)
        if epoch % 300 == 0:
            trainer.save()
    trainer.save()

else:
    for epoch in range(num_epoch):
        #print(type(trainer))
        print("Training iteration: {}".format(epoch), end='')
        res = trainer.train()
        print(f", Average reward: {res['episode_reward_mean']}")
    
        if epoch % test_epoch == 0:
            reward = env.test(trainer)
            
        if epoch % 200 == 0:
            trainer.save()


