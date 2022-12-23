import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
from matplotlib import animation
from plotter import plot_grid_figure
from useful_tools import get_img_view, get_collision_info
from models import LobulaGiantMotionDetectorModel
import copy
plt.rcParams['font.size'] = 12


# load data and world
mode = 'Dynamic'
num_obstacles = 30
data_filename = '../results/simulation_navigation_random_obs/{}Obs_T1000/LGMD/1.mat'.format(num_obstacles)
data = sio.loadmat(data_filename)
world = sio.loadmat(data['world_name'][0])

lgmd = data['lgmd_out'][0]
end_t = len(lgmd)
pos = data['position'][:end_t]
velocity = data['velocities'][:end_t]
states = data['states'][0]
h = data['headings'][0][:end_t]
pi_memory = data['pi_memory'][:end_t]
spikes = data['collision'][0]
if mode == 'Dynamic':
    collision_t = get_collision_info(world, pos, True)
else:
    collision_t = get_collision_info(world['0'][0][0], pos)

fig, ax = plot_grid_figure(9, 10,
                           [[0, 2, 1, 3],
                            [0, 2, 4, 6],
                            [0, 2, 7, 10],
                            [6, 8, 1, 4],
                            [3, 8, 5, 10],
                            [3, 5, 1, 4],
                            ], figsize=(12, 12))
# plt.tight_layout()
suptitle = 'Autonomous Navigation in {} and Low Contrast Environment  \n (Weber Contrast = 37.3%)'.format(mode,
                                                                              num_obstacles)
plt.suptitle(suptitle)
# 1. view image
img_pre = get_img_view(world['0'][0][0],
                       pos[0, 0], pos[0, 1],
                       0.01, h[0], hfov_d=100,
                       wrap=False, blur=False)
imgv = ax[0].imshow(img_pre, cmap='gray')
ax[0].set_axis_off()
ax[0].set_title('View Image')

# 2. LGMD G layer
img_lgmd_g = ax[1].imshow(img_pre, cmap='Reds')
ax[1].set_axis_off()
ax[1].set_title('LGMD: G layer')
lgmd_agent = LobulaGiantMotionDetectorModel(img_pre.shape[1],
                                            img_pre.shape[0])
lgmd_agent.reset()
lgmd_agent.frame[0] = img_pre

# 3.vector memory
bar_pi = ax[3].bar(range(1, 17), 0.5*np.ones(16),
                   align="center", color="orange")
ax[3].set_ylim(0, 1.1)
ax[3].set_title('Vector Working Memory',fontsize=14)
ax[3].grid(1)
ax[3].set_xlabel('neuron index', fontsize=12)
ax[3].set_ylabel('membrane potential', fontsize=12)

# 4. agent and world
## 4.1 food sites
food_sites = world['FoodSites']
food_catchment = world['FoodCatchment'][0]
for i, f in enumerate(food_sites):
    ax[4].add_patch(
        patches.Circle(
            xy=(f[0], f[1]),
            radius=food_catchment[i],
            fc='lightblue',    # facecolor
            ec='cornflowerblue',
            alpha=0.3
        )
    )
    ax[4].text(f[0]-food_catchment[i], 
                f[1], 
                'FoodSite{}'.format(i+1),
                fontsize=8)
## 4.2 obstacles
obj_verts = np.dstack([world['0'][0][0]['X'], world['0'][0][0]['Y']])
objs = PolyCollection(obj_verts, facecolor='k', edgecolor='none', alpha=1)
ax[4].add_collection(objs)
arena_boundary = world['Arena'][0]
ax[4].set_xlim(arena_boundary[0]-2, arena_boundary[1]+2)
ax[4].set_ylim(arena_boundary[2]-2, arena_boundary[3]+2)
## 4.3 arena boundaries
arena_boundary = world['Arena'][0]
margin = 4
rect = patches.Rectangle([arena_boundary[0]-2, arena_boundary[2]-2],
                         arena_boundary[1] - arena_boundary[0]+2,
                         arena_boundary[3] - arena_boundary[2]+2,
                         fill=False, lw=6, ls='-', ec='gray')
ax[4].add_patch(rect)
ax[4].scatter(pos[0, 0], pos[0, 1], marker='*', color="red")
## 4.4 agent
states_names = ['exploring', 'homing', 'foraging',
                'near food', 'near nest']
agent_t, = ax[4].plot(pos[0], pos[0],
                      color='orange', alpha=0.5)
agent_m, = ax[4].plot(pos[0], pos[0],
                      marker='o', color='orange', alpha=0.5)
hfov = np.deg2rad(100)
fhov_l_len = 100
hfov_l1, = ax[4].plot([pos[0, 0],
                       pos[0, 0] + fhov_l_len*np.sin(h[0]-hfov/2)],
                      [pos[0, 1],
                       pos[0, 1] + fhov_l_len*np.cos(h[0]-hfov/2)],
                      lw=1, ls='--', color='red', alpha=0.5)
hfov_l2, = ax[4].plot([pos[0, 0],
                       pos[0, 0] + fhov_l_len*np.sin(h[0]+hfov/2)],
                      [pos[0, 1],
                       pos[0, 1] + fhov_l_len*np.cos(h[0]+hfov/2)],
                      lw=1, ls='--', color='red', alpha=0.5)
ax[4].set_aspect(1)
ax[4].set_xlim(arena_boundary[0] - margin,
               arena_boundary[1] + margin)
ax[4].set_ylim(arena_boundary[2] - margin,
               arena_boundary[3] + margin)
ax[4].set_axis_off()

ax[4].scatter([], [],
              marker='o', color='orange',
              label='Agent (No Collision)', alpha=0.5)
ax[4].scatter([], [],
              marker='o', color='red',
              label='Agent (Collision)', alpha=0.5)
ax[4].legend(fontsize=10)

# 5. LGMD spike
lgmd_thr = 0.95 if mode == 'Dynamic' else 0.98
line_lgmd, = ax[2].plot(0, 0,
                        color="black", lw=0.5,
                        label='LGMD')
ax[2].plot([0, len(pos)], [lgmd_thr, lgmd_thr],
           color='gray', ls='--', lw=2,
           label='Threshold')
ax[2].set_ylim(0.4, 1.1)
ax[2].set_title('LGMD Neuron', fontsize=14)
ax[2].set_xlabel('timestep')
ax[2].set_ylabel('membrane potential')
ax[2].scatter([], [], marker='x', s=5, color='red', label='Collision')
ax[2].legend(fontsize=8, numpoints=1)

# 5. Velocity
line_vx, = ax[5].plot(0, 0,
                      color='skyblue', lw=0.8, ls='--',
                      label='Velocity_X')
line_vy, = ax[5].plot(0, 0,
                      color='tomato', lw=0.8, ls='--',
                      label='Velocity_Y')
line_v, = ax[5].plot(0, 0,
                     color='gray', lw=1,
                     label='Speed')

ax[5].set_ylim(-1.8, 1.8)
ax[5].set_title('Agent Velocity', fontsize=14)
ax[5].set_xlabel('timestep')
ax[5].set_ylabel('membrane potential')
ax[5].legend(fontsize=8)

max_t = max([int(s) for s in list(filter(lambda x:x.isnumeric(),
                                         world.keys()))]) + 1


def update(frame):
    if mode == 'Static':
        objs.set_verts(np.dstack([world['0'][0][0]['X'],
                                  world['0'][0][0]['Y'],
                        ]))
        img = get_img_view(world['0'][0][0],
                           pos[frame, 0], pos[frame, 1],
                           0.01, h[frame], hfov_d=100,
                           wrap=False, blur=False)
    else:
        # back and forth
        if frame // max_t % 2 == 0:
            ind_t = frame % max_t
        else:
            ind_t = max_t - (frame % max_t) - 1
        img = get_img_view(world['{}'.format(ind_t)][0][0],
                           pos[frame, 0], pos[frame, 1],
                           0.01, h[frame], hfov_d=100,
                           wrap=False, blur=False)
        objs.set_verts(np.dstack([world['{}'.format(ind_t)][0][0]['X'],
                                  world['{}'.format(ind_t)][0][0]['Y'],
                                ]))
    img[np.where(img >= 160)] -= 160
    imgv.set_array(img)
    lgmd_agent.frame[1] = copy.deepcopy(img)
    p, i, s, Ce, g, _, ffi = lgmd_agent.run(luminance_per_num=1,
                                            fixed_ts=True,
                                            fixed_ffi_thr=True)
    # p, i, s, Ce, g, lgmd_, ffi = lgmd_agent.run_no_enhance(luminance_per_num=1,
    #                                     fixed_ts=True,
    #                                     fixed_ffi_thr=True)
    lgmd_agent.frame[0] = copy.deepcopy(img)
    img_lgmd_g.set_array(g)
    if frame in collision_t:
        print(frame)
        ax[4].scatter(pos[frame, 0], pos[frame, 1], color='red', marker='x')
        
    if spikes[frame] == 1:
        # collision happened
        agent_m.set_color('red')
        ax[2].scatter(frame, 1.05, s=5, marker='x', color='red')
    else:
        agent_m.set_color('orange')

    line_lgmd.set_data(np.arange(frame), lgmd[:frame])
    line_vx.set_data(np.arange(frame), velocity[:frame, 0])
    line_vy.set_data(np.arange(frame), velocity[:frame, 1])
    line_v.set_data(np.arange(frame),
                    np.sqrt(velocity[:frame, 0]**2 +
                            velocity[:frame, 1]**2))
    agent_t.set_data(pos[:frame, 0], pos[:frame, 1])
    agent_m.set_data(pos[frame, 0], pos[frame, 1])

    hfov_l1.set_data([pos[frame, 0],
                      pos[frame, 0] + fhov_l_len*np.sin(h[frame]-hfov/2)],
                     [pos[frame, 1],
                      pos[frame, 1] + fhov_l_len*np.cos(h[frame]-hfov/2)])

    hfov_l2.set_data([pos[frame, 0],
                      pos[frame, 0] + fhov_l_len*np.sin(h[frame]+hfov/2)],
                     [pos[frame, 1],
                      pos[frame, 1] + fhov_l_len*np.cos(h[frame]+hfov/2)])
    
    for rect, height in zip(bar_pi, pi_memory[frame]):
        rect.set_height(height)
        rect.set_alpha(np.max([0.1, np.min([height+0.1, 1.0])]))
    
    ax[2].set_xlim([0, frame])
    ax[5].set_xlim([0, frame])
    
    ax[4].set_title('frame = {}/{}, agent state: {}'.format(frame, end_t,
                                                            states_names[
                                                                states[frame]])
                    )
    

anim = animation.FuncAnimation(fig,
                               update,
                               frames=len(pos),
                               interval=10)
plt.show()

# uncomment this line and give the save path to save as .mp4 video file
# anim.save('save_path' + '.mp4', dpi=100, writer="ffmpeg")


