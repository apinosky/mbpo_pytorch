import argparse
import time
import gym
import torch
import numpy as np
# from itertools import count

import os
# import os.path as osp
# import json

from sac.replay_memory import ReplayMemory
# from sac.sac import SAC
from sac_lib.sac import SoftActorCritic as SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
from sac_lib.normalized_actions import NormalizedActions
# from tf_models.constructor import construct_model, format_samples_for_training # commenting out all tensorflow references for lighter load

import pickle
from datetime import datetime
import logger

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env_name', default="Hopper-v2",
                        help='Mujoco Gym environment (default: Hopper-v2)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')

    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')

    # ''' SAC params '''
    # parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
    #                     help='discount factor for reward (default: 0.99)')
    # parser.add_argument('--tau', type=float, default=0.01, metavar='G',
    #                     help='target smoothing coefficient(tau) (default: 0.005)')
    # parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
    #                     help='Temperature parameter α determines the relative importance of the entropy\
    #                         term against the reward (default: 0.2)')
    # parser.add_argument('--policy', default="Gaussian",
    #                     help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    # parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
    #                     help='Value target update per no. of updates per step (default: 1)')
    # parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
    #                     help='Automaically adjust α (default: False)')
    # parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
    #                     help='hidden size (default: 256)')
    # parser.add_argument('--lr', type=float, default=0.003, metavar='G',
    #                     help='learning rate (default: 0.0003)')
    # parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
    #                     help='critic learning rate (default: 0.0003)')

    ''' predictive model ensemble params '''
    parser.add_argument('--num_networks', type=int, default=1, metavar='E',
                        help='ensemble size (default: 7)')
    parser.add_argument('--num_elites', type=int, default=1, metavar='E',
                        help='elite size (default: 5)')
    parser.add_argument('--pred_hidden_size', type=int, default=200, metavar='E',
                        help='hidden size for predictive model')
    parser.add_argument('--reward_size', type=int, default=1, metavar='E',
                        help='environment reward size')

    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')                           ## environment (sac) replay buffer size

    ''' sets model replay buff size '''
    parser.add_argument('--model_retain_epochs', type=int, default=1, metavar='A',
                        help='retain epochs')
    parser.add_argument('--model_train_freq', type=int, default=250, metavar='A',
                        help='frequency of training')
    parser.add_argument('--model_train_batch_size', type=int, default=10000, metavar='A',
                        help='rollout number M')
    parser.add_argument('--rollout_batch_size', type=int, default=100000, metavar='A',
                        help='rollout number M (orig 100000)')
    ## next 4 for adaptive rollout length, hardcoded to match hybrid learning (fixed length)
    # parser.add_argument('--rollout_min_epoch', type=int, default=20, metavar='A',
    #                     help='rollout min epoch')
    # parser.add_argument('--rollout_max_epoch', type=int, default=150, metavar='A',
    #                     help='rollout max epoch')
    # parser.add_argument('--rollout_min_length', type=int, default=1, metavar='A',
    #                     help='rollout min length')
    # parser.add_argument('--rollout_max_length', type=int, default=15, metavar='A',
    #                     help='rollout max length')
    parser.add_argument('--rollout_length', type=int, default=3, metavar='A',
                        help='rollout min epoch')                                                   ## k in algorithm 2 (replacement for 4 args above)

    ''' general simulation params '''
    parser.add_argument('--epoch_length', type=int, default=1000, metavar='A',
                        help='steps per epoch')                                                     ## E in algorithm 2
    parser.add_argument('--num_epoch', type=int, default=100, metavar='A',
                        help='total number of epochs')                                              ## N in algorithm 2 (replacing with total frames in next line)
    parser.add_argument('--max_frames', type=int, default=100000, metavar='A',
                        help='total number of epochs')                                              ## replaces N in algorithm 2 (arg above)
    parser.add_argument('--min_pool_size', type=int, default=1000, metavar='A',
                        help='minimum pool size')                                                   ## frames before learning
    parser.add_argument('--real_ratio', type=float, default=0.05, metavar='A',
                        help='ratio of env samples / model samples')
    parser.add_argument('--train_every_n_steps', type=int, default=1, metavar='A',
                        help='frequency of training policy')
    parser.add_argument('--num_train_repeat', type=int, default=20, metavar='A',
                        help='times to training policy per step')                                   ## G in algorithm 2 is num_train_repeat / epoch (so 20*1000 = 20000), 20 for hopper, 40 for cheetah?
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5, metavar='A',
                        help='max training times per step')
    parser.add_argument('--policy_train_batch_size', type=int, default=256, metavar='A',
                        help='batch size for training policy')                                      ## note: real_ratio*policy_train_batch_size pulled from env_buffer and (1-real_ratio)*policy_train_batch_size pulled from model replay buffer
    parser.add_argument('--init_exploration_steps', type=int, default=1000, metavar='A',
                        help='exploration steps initially (default 5000)')                          ## random frames before learning?

    parser.add_argument('--model_type', default='pytorch', metavar='A',
                        help='predict model -- pytorch or tensorflow')

    parser.add_argument('--cuda', type=bool, default=True,
                        help='run on CUDA (default: True)')

    return parser.parse_args()

def train(args, env_sampler, predict_env, agent, env_pool, model_pool, start):
    # save config
    with open(args.path_dir + "/config.txt","a") as f:
        f.write('\nStart Time\n')
        f.write('\t'+ start.strftime("%Y-%m-%d_%H-%M-%S/") )
        f.write('\nConfig\n')
        f.write('\t' + str(args) + '\n')
        f.close()

    total_step = 0
    reward_sum = 0
    rollout_length = args.rollout_length # 1

    exploration_before_start(args, env_sampler, env_pool, agent)
    total_step  += args.init_exploration_steps

    logger.info('starting main loop')
    epoch_step = 0
    while (total_step < args.max_frames):                                           # alg 2 step 2
    #     print(total_step, args.max_frames,total_step < args.max_frames)
    # for epoch_step in range(args.num_epoch):
        logger.info('epoch # %d',epoch_step)
        # start_step = total_step
        train_policy_steps = 0
        # logger.debug('total step | iter step | action')
        # for i in count():
        for cur_step in range(args.epoch_length):                                                                 # alg 2 step 4
            # cur_step = total_step - start_step

            # if cur_step >= start_step + args.epoch_length and len(env_pool) > args.min_pool_size:
            #     break

            # if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
            if cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                train_predict_model(args, env_pool, predict_env)                                                # alg 2 step 3

                ## following does adaptive rollout length, but commenting out since we're running with fixed length
                # new_rollout_length = set_rollout_length(args, epoch_step)
                # if rollout_length != new_rollout_length:
                #     rollout_length = new_rollout_length
                #     model_pool = resize_model_pool(args, rollout_length, model_pool)

                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)                   # alg 2 step 6/7/8

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent,total_step=total_step) # alg 2 step 5
            env_pool.push(cur_state, action, reward, next_state, done)

            if len(env_pool) > args.min_pool_size:                                                              # alg 2 step 9/10
                # if i % 100 == 0:
                #     logger.debug('%d | %d | training policy',total_step, cur_step)
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)

            total_step += 1

            # if total_step % 1000 == 0:
            #     '''
            #     avg_reward_len = min(len(env_sampler.path_rewards), 5)
            #     avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
            #     logger.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
            #     print(total_step, env_sampler.path_rewards[-1], avg_reward)
            #     '''
            #     logger.debug('%d | %d | testing policy',total_step, i)
            #     env_sampler.current_state = None
            #     sum_reward = 0
            #     done = False
            #     while not done:
            #         cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, eval_t=True, total_step=total_step)
            #         sum_reward += reward
                # print("Step Reward: " + str(total_step) + " " + str(sum_reward))
                # print(total_step, sum_reward)

            if total_step % (args.max_frames//10) == 0:
                torch.save(agent.policy_net.state_dict(), args.path_dir + 'policy_' + str(total_step) + '.pt')
            
            if total_step % (args.max_frames//100) == 0:
                env_sampler.save(args,total_step)
        epoch_step += 1
    logger.info('saving final rewards')
    env_sampler.save(args,total_step)

    # save config
    end = datetime.now()
    duration = end-start
    duration_in_s = duration.total_seconds()
    days = divmod(duration_in_s, 64*60*60)[0]
    hours = divmod(duration_in_s, 60*60)[0]
    minutes, seconds = divmod(duration_in_s, 60)
    duration_str = 'DD:HH:MM:SS {:d}:{:d}:{:d}:{:d}'.format(int(days),int(hours),int(minutes % 60),int(seconds))
    with open(args.path_dir + "/config.txt","a") as f:
        f.write('End Time\n')
        f.write('\t'+ end.strftime("%Y-%m-%d_%H-%M-%S/") + '\n')
        f.write('Duration\n')
        f.write('\t'+ duration_str + '\n')
        f.close()

def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        if i == 0:     logger.info('starting exploration before start')
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent,random_action=True)
        env_pool.push(cur_state, action, reward, next_state, done)


# def set_rollout_length(args, epoch_step):
#     rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
#                               / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
#                               args.rollout_min_length), args.rollout_max_length))
#     return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    # state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    # Get a subset of samples from environment
    state, action, reward, next_state, done = env_pool.sample(args.model_train_batch_size) #len(env_pool)
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=128, holdout_ratio=0.2)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    logger.debug('model pool size: %d',new_pool_size)

    new_model_pool = ReplayMemory(new_pool_size)
    if model_pool is not None:
        sample_all = model_pool.return_all()
        new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        if terminals is None: # use dones pulled from buff
            if i == 0:
                if len(done.shape) == 1:
                    terminals = done[:,None]
                elif len(done.shape) == 2:
                    terminals = done
                else:
                    raise ValueError('got invalid done shape',done.shape)
            else: # assume all the rest are not done (after mask)
                terminals = np.full([state.shape[0],1],False)
        # print('rollout_model output',terminals.shape)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state         = np.concatenate((env_state, model_state), axis=0)
            batch_action        = np.concatenate((env_action, model_action),axis=0)
            batch_reward        = np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0)
            batch_next_state    = np.concatenate((env_next_state, model_next_state),axis=0)
            batch_done          = np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat


# not used so commented out
# from gym.spaces import Box
# class SingleEnvWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super(SingleEnvWrapper, self).__init__(env)
#         obs_dim = env.observation_space.shape[0]
#         obs_dim += 2
#         self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
#
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
#         obs = np.append(obs, [torso_height, torso_ang])
#
#         return obs, reward, done, info
#
#     def reset(self):
#         obs = self.env.reset()
#         torso_height, torso_ang = self.env.sim.data.qpos[1:3]
#         obs = np.append(obs, [torso_height, torso_ang])
#         return obs


def main(args=None):
    # logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if args is None:
        args = readParser()

    # Save Directory
    base_path = './data/' + args.env_name + '/'
    seed ='seed_{}'.format(str(args.seed))
    args.path_dir = base_path + seed + '/'
    start = datetime.now()
    if not os.path.exists(args.path_dir):
        os.makedirs(args.path_dir)
    # logger.debug(args.path_dir)
    logger.set_file_handler(path=base_path,prefix=seed)

    # Initial environment
    env = NormalizedActions(gym.make(args.env_name).env) # remove time limit
    logger.debug(env)
    logger.debug([env.action_space.shape[0],env.observation_space.shape[0]])

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space.shape[0], args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    if args.model_type == 'pytorch':
        env_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, state_size, action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
    else:
        raise ValueError('tensorflow commented out')
        # env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks,
        #                             num_elites=args.num_elites)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type)

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    model_pool = resize_model_pool(args, args.rollout_length, model_pool=None)

    # Sampler of environment
    env_sampler = EnvSampler(env,max_path_length=args.epoch_length,seed=args.seed)

    train(args, env_sampler, predict_env, agent, env_pool, model_pool, start)
    torch.save(agent.policy_net.state_dict(), args.path_dir + 'policy_' + 'final' + '.pt')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.warn(e)

