from termcolor import cprint
import pickle
import numpy as np

import logger

class EnvSampler():
    def __init__(self, env, max_path_length=1000, seed=1234):
        self.env = env

        self.action_dim = env.action_space.shape[0]
        np.random.seed(seed)

        self.path_length = 0
        self.current_state = None
        self.max_path_length = max_path_length
        self.path_rewards = []
        self.eval_path_rewards = []
        self.sum_reward = 0
        self.ep_num = 0

    def get_random_action(self):
        return np.random.random(self.action_dim) * 2 - 1

    def sample(self, agent, eval_t=False, total_step=0, random_action=False):
        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state
        if random_action:
            action = self.get_random_action()
        else:
            action = agent.select_action(self.current_state, eval_t)
        next_state, reward, terminal, info = self.env.step(action)
        self.path_length += 1
        self.sum_reward += reward

        # TODO: Save the path to the env_pool
        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            if eval_t:
                logger.info(['eval rew', self.ep_num, self.sum_reward, self.path_length])
                self.eval_path_rewards.append([total_step,self.sum_reward,self.ep_num])
            else:
                self.ep_num += 1
                logger.info(['ep rew', self.ep_num, self.sum_reward, total_step, self.path_length])
                self.path_rewards.append([total_step,self.sum_reward,self.ep_num])
            self.path_length = 0
            self.sum_reward = 0
            terminal=True
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal, info

    def save(self,args,total_step):
        last_reward = self.path_rewards[-1][1] if len(self.path_rewards)>0 else 0
        logger.info('frame : {}/{}, \t last rew: {:.2f}'.format(total_step, args.max_frames, last_reward))
        if len(self.path_rewards) > 0:
            pickle.dump(self.path_rewards, open(args.path_dir + 'reward_data' + '.pkl', 'wb'))
        if len(self.eval_path_rewards) > 0:
            pickle.dump(self.eval_path_rewards, open(args.path_dir + 'eval_reward_data' + '.pkl', 'wb'))
