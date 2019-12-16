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


# A full free-for-all version of tron
class TronRayEnvironment(MultiAgentEnv):
    action_space = Discrete(3)

    def __init__(self, board_size=15, num_players=4):
        self.env = TronGridEnvironment.create(board_size=board_size, num_players=num_players)
        self.state = None
        self.players = None

        self.renderer = TronRender(board_size, num_players)

        self.observation_space = Dict({
            'board': Box(0, num_players, shape=(board_size, board_size)),
            'heads': Box(0, np.infty, shape=(num_players,)),
            'directions': Box(0, 4, shape=(num_players,)),
            'deaths': Box(0, num_players, shape=(num_players,))
        })

    def reset(self):
        self.state, self.players = self.env.new_state()
        return {str(i): self.env.state_to_observation(self.state, i) for i in range(self.env.num_players)}

    def step(self, action_dict):
        #print("CALLING STEP ****************************************************************************************")
        action_to_string = {
            0: 'forward',
            1: 'right',
            2: 'left'
        }
        

        actions = []

        for player in self.players:
            action = action_dict.get(str(player), 0)
            actions.append(action_to_string[action])
            

        self.state, self.players, rewards, terminal, winners = self.env.next_state(self.state, self.players, actions)

        num_players = self.env.num_players
        alive_players = set(self.players)

        observations = {str(i): self.env.state_to_observation(self.state, i) for i in map(int, action_dict.keys())}
        rewards = {str(i): rewards[i] for i in map(int, action_dict.keys())}
        dones = {str(i): i not in alive_players for i in map(int, action_dict.keys())}
        dones['__all__'] = terminal
        if dones['0'] == True:
            dones['1'] = True
            dones['2'] = True
            dones['3'] = True
            dones["__all__"] = True
            

        return observations, rewards, dones, {}

    def render(self, mode='human'):
        if self.state is None:
            return None

        return self.renderer.render(self.state, mode)

    def close(self):
        self.renderer.close()
        
    def test(self, trainer, frame_time = 0.1):
        num_players = self.env.num_players
        self.close()
        state = self.reset()
        done = {"__all__": False}
        action = {str(i): None for i in range(num_players)}
        reward = {str(i): None for i in range(num_players)}
        cumulative_reward = 0
        
        while not done['__all__']:
            action = {i: trainer.compute_action(state[i], prev_action=action[i], prev_reward=reward[i], policy_id="opponent") for
                      i in map(str, range(num_players))}
            action['0'] = trainer.compute_action(state['0'], prev_action=action['0'], prev_reward=reward['0'], policy_id="trainer")
            
            state, reward, done, results = self.step(action)
            cumulative_reward += sum(reward.values())
            if done['0'] == True:
                print("Player Died")
            self.render()
            
            sleep(frame_time)

        
        self.render()
        return cumulative_reward



# Some preprocessing to let the networks learn faster
class TronExtractBoard(Preprocessor):
    def _init_shape(self, obs_space, options):
        board_size = env.observation_space['board'].shape[0]
        self.rotate_counter = 0
        return (board_size + 4, board_size + 4, 2)
    
    def transform(self, observation):
        self.rotate_counter
        if self.rotate_counter == 4:
            self.rotate_counter = 0
        self.rotate_counter += 1
        #print(observation)
        rotate = self.rotate_counter + 1
        if rotate == 5:
            rotate = 1
        new_board = self._transform(observation,rotate=rotate)
        return new_board
        '''
        if 'board' in observation:
            print("******************************************************************************************************")
            #print(player_num)
            return self._transform(observation)
        else:
            #print("************************************************************************************")
            return {key: self._transform(value) for key, value in observation.items()}
        '''
    
    def _transform(self, observation, rotate: int = 0):
        board = observation['board'].copy()
        
        # Make all enemies look the same
        board[board > 1] = -1
        
        # Mark where all of the player heads are
        heads = np.zeros_like(board)
        
        if (rotate != 0):
            heads.ravel()[observation['heads']] += 1 + ((observation['directions'] - rotate) % 4)
           
            board = np.rot90(board, k=rotate)
            heads = np.rot90(heads, k=rotate)
           
        else:
            heads.ravel()[observation['heads']] += 1 + observation['directions']
        
        # Pad the outsides so that we know where the wall is
        board = np.pad(board, 2, 'constant', constant_values=-1)
        heads = np.pad(heads, 2, 'constant', constant_values=-1)
        
        # Combine together
        board = np.expand_dims(board, -1)
        heads = np.expand_dims(heads, -1)
        
        return np.concatenate([board, heads], axis=-1)


# Initialize training environment
ray.init()

def environment_creater(params=None):
    return TronRayEnvironment(board_size=13, num_players=4)

env = environment_creater()
tune.register_env("tron_multi_player", environment_creater)
ModelCatalog.register_custom_preprocessor("tron_prep", TronExtractBoard)

# Configure Deep Q Learning for multi-agent training
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
#config["timesteps_per_iteration"] = 128
#config['target_network_update_freq'] = 64
#config['buffer_size'] = 100_000
#config['schedule_max_timesteps'] = 10_000
#config['exploration_fraction'] = 0.02
#config['compress_observations'] = False
#config['n_step'] = 2

# All of the models will use the same network as before
agent_config = {
    "model": {
        "vf_share_layers": True,
        "conv_filters": [(512, 5, 1), (256, 3, 2), (128, 3, 2)],
        "fcnet_hiddens": [256],
        "custom_preprocessor": 'tron_prep'
    }
}

def policy_mapping_function(x):
    if x == '0':
        return "trainer"
    return "opponent"

config['multiagent'] = {
        "policy_mapping_fn": policy_mapping_function,
        "policies": {"trainer": (None, env.observation_space, env.action_space, agent_config), "opponent":(None, env.observation_space, env.action_space, agent_config)},
        "policies_to_train":["trainer"]
}
       
trainer = PPOTrainer(config, "tron_multi_player")
#trainer.restore("./desktop_version/checkpoint_1802/checkpoint-1802")
trainer.restore("./ppo_selfplay/sp_checkpoint_2257/checkpoint-2257")

num_epoch = 1000
save_epochs = 50
update_times = 0
#update_percentage = update_times * 0.01
epoch_update = 0

for epoch in range(num_epoch):
    print("Training iteration: {}".format(epoch), end='\t')
    res = trainer.train()
    win_percentage = (res["policy_reward_mean"]["trainer"] - res["episode_len_mean"])/11 - 10/11 + 1
    print("Win percentage: ", win_percentage, end='\t')
    print("Average reward: ", res["policy_reward_mean"]["trainer"] )
    update_percentage = update_times * 0.01
    if win_percentage > 0.72 + update_percentage or win_percentage > 0.82:
#    and res["policy_reward_mean"]["trainer"] > 18 + update_times:
        if epoch_update == 0:
            epoch_update = epoch
        
        if epoch >= epoch_update + 5:
            update_times += 1
            epoch_update = epoch
            print("UPDATING OPPONENTS")
            trainer_weights = trainer.get_policy("trainer").get_weights()
            trainer.get_policy("opponent").set_weights(trainer_weights)
            reward = env.test(trainer)
    if epoch % save_epochs == 0:
        trainer.save()
    #print(res)
    #print("Average reward: ", res["policy_reward_mean"]["trainer"] )
    
    
    
    if epoch % 1 == 0:
       reward = env.test(trainer)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
