import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection

from useful_tools import get_next_state, get_img_view, get_collision_info

import copy
import scipy.io as sio
from plotter import plot_grid_figure

from agent import NavigationAgent

plt.rcParams['font.size'] = 12

world_name = 'worlds/3FoodS200_Trans/30Obs_T100.mat'

world = sio.loadmat(world_name)
dyn_max_t = max([int(s) for s in list(filter(lambda x:x.isnumeric(),
                                             world.keys()))]) + 1
agent = NavigationAgent(world)

fig, ax = plot_grid_figure(3, 3,
                           [[0, 1, 0, 1],
                            [0, 1, 1, 2],
                            [0, 1, 2, 3],
                            [2, 3, 0, 1],
                            [1, 3, 1, 3],
                            [1, 2, 0, 1],
                            ])
T = 20000

pos = [[world['StartPos'][0][0],
        world['StartPos'][0][1]]]
h = [np.random.vonmises(0.0, 100.0, 1)[0]*0.5]
velocity = [[0., 0.]]
lgmd = []
ffi = []
agent.hfov_d=120
img_pre = get_img_view(world['0'][0][0],
                       pos[0][0], pos[0][1],
                       0.01, h[0], hfov_d=agent.hfov_d,
                       wrap=False, blur=False)
agent.lgmd.img_height = img_pre.shape[0]
agent.lgmd.img_width = img_pre.shape[1]
agent.lgmd.reset()
agent.lgmd.frame[0] = img_pre
# LGMD parameter - static
# agent.lgmd.lgmd_thr = 0.99
# agent.lgmd.ffi_thr = 0.2
# agent.lgmd.spike_interval = 10
# agent.lgmd.i_w = 0.35

# LGMD parameter - dynamic
agent.lgmd.lgmd_thr = 0.96 # 0.96
agent.lgmd.ffi_thr = 0.2
agent.lgmd.spike_interval = 5 # 8
agent.lgmd.i_w = 0.35 

# image views
## input image
img_ = ax[0].imshow(img_pre, cmap='gray')
ax[0].set_axis_off()
ax[0].set_title('Image Reconstructed')
img_g = ax[1].imshow(img_pre, cmap='gray')
ax[1].set_axis_off()
ax[1].set_title('G-layer of LGMD')
# obstacles
static_world_t = 'None'
if not static_world_t == 'None':
    obj_verts = np.dstack([world[static_world_t][0][0]['X'],
                           world[static_world_t][0][0]['Y']])
else:
    obj_verts = np.dstack([world['0'][0][0]['X'],
                           world['0'][0][0]['Y']])
objs = PolyCollection(obj_verts, facecolor='k', edgecolor='none', alpha=1)
ax[4].add_collection(objs)
boundary = world['Arena'][0]
ax[4].set_xlim(boundary[0]-2, boundary[1]+2)
ax[4].set_ylim(boundary[2]-2, boundary[3]+2)
food_sites = world['FoodSites']
food_radius = world['FoodCatchment'][0]
# food sites
for i, f in enumerate(food_sites):
    ax[4].add_patch(
        patches.Circle(
            xy=(f[0], f[1]),
            radius=food_radius[i],
            fc='white',    # facecolor
            ec='cornflowerblue',
        )
    )

# lgmd model outputs
## lgmd value and spikes
lgmd_v, = ax[3].plot(0, 0, color='orange', lw=0.5, label='LGMD')
lgmd_thr_l, = ax[3].plot(0, 0, color='orange', ls='-.', lw=0.5)
lgmd_s, = ax[5].plot(0, 0, color='orange', lw=0.5, label='LGMD')
## ffi value
ffi_v, = ax[3].plot(0, 0, color='skyblue', lw=0.5, label='FFI')
ffi_thr_l, = ax[3].plot(0, 0, color='skyblue', ls='-.', lw=0.5)
ffi_s, = ax[5].plot(0, 0, color='skyblue', lw=0.5, label='FFI')
## collision
collision, = ax[5].plot(0, 0, color='red', lw=0.5)

ax[3].set_ylim(-0.1, 1.1)
ax[3].set_title('Collision Avoidance-LGMD/FFI',fontsize=12)
ax[3].grid(1)
ax[3].set_xlabel('timestep', fontsize=12)
ax[3].set_ylabel('membrane potential', fontsize=12)
ax[3].legend(fontsize=8)
ax[5].set_title('Collision Avoidance-LGMD/FFI Spikes')
ax[5].set_ylim(-0.1, 1.1)
ax[5].grid(1)
ax[5].set_xlabel('timestep', fontsize=12)
ax[5].set_ylabel('Sipikes', fontsize=12)
ax[5].legend(fontsize=8)

# agent move in anera
agent_t, = ax[4].plot(pos[0][0], pos[0][1],
                      color='green', alpha=0.5)
agent_m, = ax[4].plot(pos[0][0], pos[0][1],
                      marker='o', color='black',
                      label='Agent', alpha=0.5)
ax[4].scatter(pos[0][0], pos[0][1], 
              marker='*', color="red",
              label='start position')


fhov_l_len = 100

hfov = np.deg2rad(agent.hfov_d)
hfov_l1, = ax[4].plot([pos[0][0],
                       pos[0][0] + fhov_l_len*np.sin(h[0]-hfov/2)],
                      [pos[0][1],
                       pos[0][1] + fhov_l_len*np.cos(h[0]-hfov/2)],
                      lw=1, ls='--', color='red', alpha=0.5)

hfov_l2, = ax[4].plot([pos[0][0],
                       pos[0][0] + fhov_l_len*np.sin(h[0]+hfov/2)],
                      [pos[0][1],
                       pos[0][1] + fhov_l_len*np.cos(h[0]+hfov/2)],
                      lw=1, ls='--', color='red', alpha=0.5)
ax[4].set_aspect(1)
ax[4].legend(fontsize=8)

pi_memory = [np.zeros([16])]
collision_detection = True

# Initialise TB and memory
tb1 = np.zeros(8)
memory = 0.5 * np.ones(16)

line_vx, = ax[2].plot(0, 0,
                      color='skyblue', lw=0.8, ls='--',
                      label='Velocity_X')
line_vy, = ax[2].plot(0, 0,
                      color='tomato', lw=0.8, ls='--',
                      label='Velocity_Y')
line_v, = ax[2].plot(0, 0,
                     color='gray', lw=1,
                     label='Speed')

ax[2].set_ylim(-1.8, 1.8)
ax[2].set_title('Agent Velocity', fontsize=14)
ax[2].set_xlabel('timestep')
ax[2].set_ylabel('step/timestep')
ax[2].legend(fontsize=8)


# otherwise, update will be called twice, so weired
def init():
    #do nothing
    pass

def set_forage_init():
    agent.vector_memories = np.array([[0.62071253, 0.61697394, 0.46099266, 0.39479437, 0.38040711,
         0.39628539, 0.47006988, 0.6162459 , 0.62110611, 0.61683387,
         0.46658736, 0.40143151, 0.38615459, 0.40313934, 0.47277713,
         0.61647089], [0.37982967, 0.27398098, 0.34790703, 0.44653348, 0.64171669,
         0.78988793, 0.59676467, 0.43543574, 0.39267553, 0.28450108,
         0.36131821, 0.45382464, 0.63786943, 0.7870762 , 0.58646682,
         0.43613251], [0.33072209, 0.68405361, 0.47610648, 0.77904476, 0.56400732,
         0.36419556, 0.41413717, 0.16362743, 0.35321689, 0.69506059,
         0.46122617, 0.75905833, 0.57104543, 0.38402494, 0.44847491,
         0.20928389]])

    agent.food_found_index = [0, 1, 2]
    agent.has_explored = True
    agent.current_state = 2


set_forage_init()

agent.cx.noise=0.1


def update(t):
    global tb1, memory

    if static_world_t == 'None':
        # back and forth
        if t // dyn_max_t % 2 == 0:
            ind_t = t % dyn_max_t
        else:
            ind_t = dyn_max_t - (t % dyn_max_t) - 1
        img = get_img_view(world[str(ind_t)][0][0],
                            pos[t][0], pos[t][1],
                            0.01, h[t], hfov_d=agent.hfov_d,
                            wrap=False, blur=False)
        objs.set_verts(np.dstack([world['{}'.format(ind_t)][0][0]['X'],
                                  world['{}'.format(ind_t)][0][0]['Y']]))
    else:
        img = get_img_view(world[static_world_t][0][0],
                           pos[t][0], pos[t][1],
                           0.01, h[t], hfov_d=agent.hfov_d,
                           wrap=False, blur=False)

    agent.lgmd.frame[1] = copy.deepcopy(img)
    p, i, s, Ce, g, _, ffi_ = agent.lgmd.run(luminance_per_num=1,
                                               fixed_ts=True,
                                               fixed_ffi_thr=True)
    agent.lgmd.frame[0] = copy.deepcopy(img)
    # CX path integration
    tl2, cl1, tb1, tn1, tn2, memory, \
        cpu4, cpu1, pi_motor = agent.update_cells(
            heading=h[t],
            velocity=np.array(velocity)[t],
            tb1=tb1,
            memory=memory)
    # inner state driven desired moving direction
    if agent.current_state == 0:
        # explore mode - randome walking
        motor = np.random.vonmises(0.0, 100.0, 1)[0]*0.5
    elif (agent.current_state == 3) or (agent.current_state == 4):
            # near-range sensor driving
            goal_dir = np.arctan2(agent.near_goal_pos[0]-pos[t][0],
                                  agent.near_goal_pos[1]-pos[t][1])
            motor = goal_dir - h[t]
    else:
        # homing using vector navigation
        motor = pi_motor
    
    if (agent.boundary_counter == 0) and (agent.collision_counter == 0):
        # if at the boundary
        if (pos[t][0] < boundary[0]) or (pos[t][0] > boundary[1]) \
            or (pos[t][1] < boundary[2]) or (pos[t][1] > boundary[3]):
                h_tmp = h[t].copy()
                v_tmp = [0, 0]
                agent.boundary_counter = 5
        else:
            if (agent.lgmd.spikes[-1] == 1) and (collision_detection):
                h_tmp = h[t].copy()
                v_tmp = [0, 0]
                agent.collision_turn_flag = (np.argmax(g) % agent.lgmd.img_width) >\
                    agent.lgmd.img_width/2
                agent.collision_counter = 2
            else:
                h_tmp, v_tmp = get_next_state(
                    h[t], np.array(velocity[t]), motor,
                    acceleration=0.4, drag=0.2)
    else:
        h_tmp = h[t].copy()
        v_tmp = velocity[t].copy()
        if agent.boundary_counter > 0:
            if agent.boundary_counter >= 5:
                h_tmp, v_tmp = get_next_state(
                            h[t], np.array(velocity[t]),
                            3*np.pi/4,
                            acceleration=0.4, drag=0.2)
            agent.boundary_counter -= 1
        if agent.collision_counter > 0:
            agent.collision_counter -= 1
    h.append(h_tmp)
    velocity.append([v_tmp[0], v_tmp[1]])
    pos.append([pos[t][0] + v_tmp[0],
                pos[t][1] + v_tmp[1]])
    lgmd.append(agent.lgmd.lgmd)
    ffi.append(agent.lgmd.ffi)
    # state calculation
    if agent.current_state == 0:
        # exploring
        for i, f in enumerate(food_sites):
            if i not in agent.food_found_index:
                if np.sqrt((pos[t+1][0] - f[0])**2 +
                           (pos[t+1][1] - f[1])**2) <= food_radius[i]:
                    # near reward
                    print('''***t={}:near food {}.'''.format(t, i+1))
                    agent.current_state = 3
                    agent.food_found_index.append(i)
                    agent.near_goal_pos = f.copy()
                    break

    elif agent.current_state == 1:
        # homing
        if np.sqrt((pos[t+1][0] - pos[0][0])**2 +
                   (pos[t+1][1] - pos[1][1])**2) <= 10:
            # near home
            agent.current_state = 4
            agent.near_goal_pos = pos[0].copy()
            
    elif agent.current_state == 2:
        # foraging
        if agent.foraging_initialized:
            ind = agent.food_found_index[agent.current_forage_food_index]
            dis = np.sqrt((pos[t+1][0] - food_sites[ind][0])**2 +
                            (pos[t+1][1] - food_sites[ind][1])**2)
            if dis <= 1:
                print('''***t={}:got food {} via vector memory foraging'''.format(t,
                                                                                  ind + 1))
                agent.foraged_food_index.append(agent.current_forage_food_index)
                agent.foraged_food_time.append(t)
                if len(agent.foraged_food_index) == len(food_sites):
                    agent.current_state = 1
                    print('***t={}:got last food by vector memory, homing...'.format(t))
                    memory = agent.vector_memories[agent.current_forage_food_index].copy()
                    tb1 = np.zeros(8)
                    agent.cx.cpu4_mem_gain = 0.005
                else:
                    # got food, get next food site as target
                    diff = 100
                    min_ind = None
                    for i, f_m in enumerate(agent.vector_memories):
                        if i not in agent.foraged_food_index:
                            tmp = np.sum(np.abs(f_m - memory))
                            if tmp < diff:
                                diff = copy.copy(tmp)
                                min_ind = copy.copy(i)
                    memory += agent.vector_memories[
                        agent.current_forage_food_index] - \
                        agent.vector_memories[min_ind]
                    memory = np.clip(memory, 0, 1)
                    tb1 = np.zeros(8)
                    agent.cx.cpu4_mem_gain = 0.005
                    agent.current_forage_food_index = copy.copy(min_ind)
                    print('***Start foraging food site {}'.format(
                        agent.food_found_index[min_ind]
                        + 1))
            elif dis <= food_radius[ind]:
                agent.current_state = 3
                agent.near_goal_pos = food_sites[ind].copy()
        else:
            # select the closed food site as the first target
            diff = []
            for f_m in agent.vector_memories:
                diff.append(np.sum(np.abs(f_m - memory)))
            ind = np.argmin(np.array(diff))
            agent.current_forage_food_index = copy.copy(ind)
            # inverse vector memory
            memory = 1.0 - agent.vector_memories[ind]
            tb1 = np.zeros(8)
            agent.foraging_initialized = True
            agent.cx.cpu4_mem_gain = 0.005
            print('***Memory based foraging start with food {}'.format(
                agent.food_found_index[ind] + 1))
    elif agent.current_state == 3:
        # near food
        if np.sqrt((pos[t+1][0] - agent.near_goal_pos[0])**2 +
                    (pos[t+1][1] - agent.near_goal_pos[1])**2) <= 1:
            if agent.has_explored:
                agent.current_state = 2
            else:
                # got food
                agent.current_state = 1
                agent.step_found_food.append(t)
                agent.vector_memories.append(memory.copy())
                agent.pi_memory_snapshot = memory.copy()
                remain = list(set(range(len(food_sites)))
                                - set(agent.food_found_index))
                print('***t={}:got food {}, remain: {} .'.format(
                    t, agent.food_found_index[-1]+1, np.array(remain)+1))
    elif agent.current_state == 4:
        # near nest
        if np.sqrt((pos[t+1][0] - pos[0][0])**2 +
                    (pos[t+1][1] - pos[0][1])**2) <= 1:
            if agent.has_explored:
                print('***t={}:task completed'.format(t))
                agent.forage_completed = True
                return
            else:
                # record the timestamp of getting nest
                agent.step_got_home.append(t)
                # recaliborate the vector memory
                agent.vector_memories[-1] - memory + 0.5
                print('***t={}:got nest'.format(t))
                # got all food, exit
                if len(agent.food_found_index) == len(food_sites):
                    agent.has_explored = True
                    return
                # got original site (home)
                agent.current_state = 0
                memory = np.ones(16)*0.5
                agent.cx.cpu4_mem_gain = 0.005
    else:
        pass
    
    agent_t.set_data(np.array(pos)[:, 0],
                     np.array(pos)[:, 1])
    
    agent_m.set_data(pos[t][0], pos[t][1])
    
    hfov_l1.set_data([pos[t][0],
                      pos[t][0] + fhov_l_len*np.sin(h[t]-hfov/2)],
                     [pos[t][1],
                      pos[t][1] + fhov_l_len*np.cos(h[t]-hfov/2)])

    hfov_l2.set_data([pos[t][0],
                      pos[t][0] + fhov_l_len*np.sin(h[t]+hfov/2)],
                     [pos[t][1],
                      pos[t][1] + fhov_l_len*np.cos(h[t]+hfov/2)])
    
    if agent.lgmd.spikes[t] == 1:
        # collision happened
        agent_m.set_color('red')
    else:
        agent_m.set_color('royalblue')
    
    img_.set_array(img)
    img_g.set_array(s)
    # LGMD
    lgmd_v.set_data(np.arange(t), np.array(lgmd)[:t])
    lgmd_s.set_data(np.arange(len(agent.lgmd.lgmd_spikes)), agent.lgmd.lgmd_spikes)
    # FFI
    ffi_v.set_data(np.arange(t), np.array(ffi)[:t])
    ffi_s.set_data(np.arange(len(agent.lgmd.ffi_spikes)), agent.lgmd.ffi_spikes)
    
    ax[5].set_xlim([0, t])
    ax[3].set_xlim([0, t])
    ax[2].set_xlim([0, t])
    
    line_vx.set_data(np.arange(t), np.array(velocity)[:t, 0])
    line_vy.set_data(np.arange(t), np.array(velocity)[:t, 1])
    line_v.set_data(np.arange(t),
                    np.sqrt(np.array(velocity)[:t, 0]**2 +
                            np.array(velocity)[:t, 1]**2))
    
    # print(np.array(lgmd))


anim = animation.FuncAnimation(fig,
                               update,
                               init_func=init,
                               frames=T-1,
                               interval=10)


plt.show()

# for i in range(100):
#     update(i)


if static_world_t == 'None':
    print(get_collision_info(world, np.array(pos), dynamic=True))
else:
    print(get_collision_info(world[static_world_t][0][0],
                             np.array(pos),
                             dynamic=False))