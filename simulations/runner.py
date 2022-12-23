import scipy.io as sio
import numpy as np
import os

from useful_tools import mkdir
from agent import NavigationAgent

phase1_data = sio.loadmat('Phase1.mat')

world_name = phase1_data['world_name'][0]
vector_memory = phase1_data['vector_memories']
food_founds = phase1_data['food_found_index'][0]


def run_static(n_start, n_end, obstacles_nums):
    obstacles_period = 100
    num_trials = n_end
    agent = NavigationAgent()

    T = 20000

    for obs_num in obstacles_nums:
        world_filename_p2 = 'worlds/3FoodS200_Trans' +\
            '/{}Obs_T{}.mat'.format(obs_num,
                                    obstacles_period)
        world = sio.loadmat(world_filename_p2)
        for use_lgmd in [True, False]:
            s = 'Start: {} obstacles move in T={}. LGMD-{}'
            print(s.format(obs_num,
                           obstacles_period,
                           use_lgmd))
            n = n_start
            while n < num_trials:
                print('*Trial {}...'.format(n))
                agent.set_foraging_initialized(vector_memory, food_founds)
                start_h = np.random.vonmises(0.0, 100.0, 1)[0]
                pos, h, velocity, pi_memory, \
                    lgmd_mp, ffi_mp, states = agent.run(world,
                                                        start_h,
                                                        T,
                                                        dynamic_world=False,
                                                        collision_detection=use_lgmd,
                                                        )
                if agent.forage_completed:
                    save_dic = {
                        'world_name': world_filename_p2,
                        'position': pos, 'velocities': velocity,
                        'headings': h, 'states': states,
                        'pi_memory': pi_memory,
                        'found_food_time': agent.foraged_food_time,
                        'collision': agent.lgmd.spikes,
                        'lgmd_out': lgmd_mp, 'ffi_out': ffi_mp,
                    }
                    dir, f = os.path.split(world_filename_p2)
                    save_dir = dir.replace('worlds', 'results') + '/final/' + f[:-4] + '/Static'
                    if use_lgmd:
                        save_dir += '/LGMD'
                    else:
                        save_dir += '/NoLGMD'
                    mkdir(save_dir)
                    sio.savemat(save_dir + '/{}.mat'.format(n), save_dic)
                    n += 1
                else:
                    if not agent.out_arena:
                        n += 1


def run_dynamic(n_start, n_end, obstacles_nums):
    obstacles_period = 100
    num_trials = n_end
    agent = NavigationAgent()

    T = 20000

    for obs_num in obstacles_nums:
        # world_filename_p2 = world_name +\
        #     '/{}Obs_T{}.mat'.format(obs_num,
        #                             obstacles_period)
        # world_filename_p2 = 'worlds/3FoodS200_Trans' +\
        #     '/{}Obs_T{}.mat'.format(obs_num,
        #                             obstacles_period)
        
        world_filename_p2 = 'worlds/3FoodS200_Trans' +\
            '/{}Obs_T{}.mat'.format(obs_num,
                                    obstacles_period)
        world = sio.loadmat(world_filename_p2)
        for use_lgmd in [True]:
            s = 'Start: {} obstacles move in T={}. LGMD-{}'
            print(s.format(obs_num,
                           obstacles_period,
                           use_lgmd))
            n = n_start
            while n < num_trials:
                print('*Trial {}...'.format(n))
                agent.set_foraging_initialized(vector_memory, food_founds)
                start_h = np.random.vonmises(0.0, 100.0, 1)[0]
                pos, h, velocity, pi_memory, \
                    lgmd_mp, ffi_mp, states = agent.run(world,
                                                        start_h,
                                                        T,
                                                        dynamic_world=True,
                                                        collision_detection=use_lgmd,
                                                        )
                if agent.forage_completed:
                    save_dic = {
                        'world_name': world_filename_p2,
                        'position': pos, 'velocities': velocity,
                        'headings': h, 'states': states,
                        'pi_memory': pi_memory,
                        'found_food_time': agent.foraged_food_time,
                        'collision': agent.lgmd.spikes,
                        'lgmd_out': lgmd_mp, 'ffi_out': ffi_mp,
                    }
                    dir, f = os.path.split(world_filename_p2)
                    # save_dir = dir.replace('worlds', 'results') + '/final/' + f[:-4] + '/Dynamic'
                    save_dir = dir.replace('worlds', 'results') + '/low_contrast/' + f[:-4]
                    if use_lgmd:
                        save_dir += '/LGMD'
                    else:
                        save_dir += '/NoLGMD'
                    mkdir(save_dir)
                    sio.savemat(save_dir + '/{}.mat'.format(n), save_dic)
                    n += 1
                else:
                    if not agent.out_arena:
                        n += 1

def run_dynamic_lgmd_contrast(n_start, n_end, obstacles_nums, contrast_decay):
    obstacles_period = 100
    num_trials = n_end
    agent = NavigationAgent()

    T = 20000

    for obs_num in obstacles_nums:
        world_filename_p2 = 'worlds/3FoodS200_Trans' +\
            '/{}Obs_T{}.mat'.format(obs_num,
                                    obstacles_period)
        world = sio.loadmat(world_filename_p2)
        for c_d in contrast_decay:
            s = 'Start: {} obstacles move in T={}. Contrast-{}'
            print(s.format(obs_num,
                           obstacles_period,
                           255-c_d))
            n = n_start
            while n < num_trials:
                print('*Trial {}...'.format(n))
                agent.set_foraging_initialized(vector_memory, food_founds)
                start_h = np.random.vonmises(0.0, 100.0, 1)[0]
                pos, h, velocity, pi_memory, \
                    lgmd_mp, ffi_mp, states = agent.run(world,
                                                        start_h,
                                                        T,
                                                        dynamic_world=True,
                                                        collision_detection=True,
                                                        c_decay=c_d
                                                        )
                if agent.forage_completed:
                    save_dic = {
                        'world_name': world_filename_p2,
                        'position': pos, 'velocities': velocity,
                        'headings': h, 'states': states,
                        'pi_memory': pi_memory,
                        'found_food_time': agent.foraged_food_time,
                        'collision': agent.lgmd.spikes,
                        'lgmd_out': lgmd_mp, 'ffi_out': ffi_mp,
                    }
                    dir, f = os.path.split(world_filename_p2)
                    save_dir = dir.replace('worlds', 'results') + '/low_contrast/' + f[:-4]
                    save_dir += '/C{}'.format(255 - c_d)
                    mkdir(save_dir)
                    sio.savemat(save_dir + '/{}.mat'.format(n), save_dic)
                    n += 1
                else:
                    if not agent.out_arena:
                        n += 1

def run_dynamic_lgmd_no_enhance(n_start, n_end, obstacles_nums):
    obstacles_period = 100
    num_trials = n_end
    agent = NavigationAgent()

    T = 20000

    for obs_num in obstacles_nums:
        world_filename_p2 = 'worlds/3FoodS200_Trans' +\
            '/{}Obs_T{}.mat'.format(obs_num,
                                    obstacles_period)
        world = sio.loadmat(world_filename_p2)
        
        s = 'Start: {} obstacles move in T={}'
        print(s.format(obs_num,
                        obstacles_period))
        n = n_start
        while n < num_trials:
            print('*Trial {}...'.format(n))
            agent.set_foraging_initialized(vector_memory, food_founds)
            start_h = np.random.vonmises(0.0, 100.0, 1)[0]
            pos, h, velocity, pi_memory, \
                lgmd_mp, ffi_mp, states = agent.run(world,
                                                    start_h,
                                                    T,
                                                    dynamic_world=True,
                                                    collision_detection=True,
                                                    )
            if agent.forage_completed:
                save_dic = {
                    'world_name': world_filename_p2,
                    'position': pos, 'velocities': velocity,
                    'headings': h, 'states': states,
                    'pi_memory': pi_memory,
                    'found_food_time': agent.foraged_food_time,
                    'collision': agent.lgmd.spikes,
                    'lgmd_out': lgmd_mp, 'ffi_out': ffi_mp,
                }
                dir, f = os.path.split(world_filename_p2)
                save_dir = dir.replace('worlds', 'results') + '/lgmd_no_enhance1/' + f[:-4]
                mkdir(save_dir)
                sio.savemat(save_dir + '/{}.mat'.format(n), save_dic)
                n += 1
            else:
                if not agent.out_arena:
                    n += 1
                        
if __name__ == "__main__":
    # run_dynamic(0, 10, range(30, 42, 6))
    # run_dynamic(20, 40, [36])
    # run_dynamic_lgmd_contrast(0, 20, [30], range(220,260,20))
    run_dynamic_lgmd_no_enhance(0, 15, range(6, 48, 6))