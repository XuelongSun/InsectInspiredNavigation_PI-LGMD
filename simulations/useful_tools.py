
import io
import os

import cv2
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from scipy import signal as s_signal
from scipy.interpolate import interp1d
import scipy.io as sio

def pi2pi(theta):
    """Constrains value to lie between -pi and pi."""
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi


def cart2sph(X, Y, Z):
    """Converts cartesian to spherical coordinates.
    Works on matrices so we can pass in e.g. X with rows of len 3 for polygons."""

    XY = X ** 2 + Y ** 2
    TH = np.arctan2(X, Y)  # theta: azimuth
    PHI = np.arctan2(Z, np.sqrt(XY))  # phi: elevation from XY plane up
    R = np.sqrt(XY + Z ** 2)  # r

    return (TH, PHI, R)


def get_next_state(heading, velocity, rotation, acceleration, drag=0.5):
    """Get new heading and velocity, based on relative rotation and
    acceleration and linear drag."""
    theta = (heading + rotation + np.pi) % (2.0 * np.pi) - np.pi
    v = velocity + np.array([np.sin(theta), np.cos(theta)]) * acceleration
    v -= drag * v
    return theta, v


def generate_random_route(T=1000, vary_speed=True, drag=0.15,
                          mean_acc=0.15, max_acc=0.15, min_acc=0.0,
                          start_position=np.array([0,0])):
    # Generate random turns
    mu = 0.0
    vm = np.random.vonmises(mu, 100.0, T)
    rotation = s_signal.lfilter([1.0], [1, -0.4], vm)
    rotation[0] = 0.0

    # Randomly sample some points within acceptable acceleration and
    # interpolate to create smoothly varying speed.
    if vary_speed:
        if T > 200:
            num_key_speeds = int(T / 50)
        else:
            num_key_speeds = 4
        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T, endpoint=True)
        acceleration = f(xnew)
    else:
        acceleration = mean_acc * np.ones(T)

    # Get headings and velocity for each step
    headings = np.zeros(T)
    velocity = np.zeros([T, 2])

    for t in range(1, T):
        headings[t], velocity[t, :] = get_next_state(
            heading=headings[t-1], velocity=velocity[t-1, :],
            rotation=rotation[t], acceleration=acceleration[t], drag=drag)
    
    # Get positions
    xy = np.vstack([start_position, np.cumsum(velocity, axis=0)])
    x, y = xy[:, 0], xy[:, 1]
    
    return headings, velocity, x, y


def img_wrapper(img, R_scale):
    img_h = img.shape[0]
    img_w = img.shape[1]
    R = int(img_h * R_scale)
    deta_r = img_h / float(R)
    deta_theta = 2.0 * np.pi / img_w
    img_polar = np.zeros([2 * R, 2 * R], np.uint8)
    for i in range(2 * R):
        for j in range(2 * R):
            x = i - R
            y = R - j
            r = np.sqrt(x ** 2 + y ** 2)
            if r < R:
                theta = np.arctan2(y, x)
                #         print r, theta
                m = int(np.rint(r * deta_r))
                n = int(np.rint(theta / deta_theta))
                if n >= img_w:
                    n = img_w - 1
                if m >= img_h:
                    m = img_h - 1
                img_polar[i, j] = img[m, n]
    return img_polar


def get_img_view(world, x, y, z, th, res=1, hfov_d=360, v_max=np.pi / 2, v_min=-np.pi / 12,
                 wrap=False, blur=False, blur_kernel_size=3):
    """Get view from the simulated 3D world

    Args:
        world (_type_): _description_
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        th (_type_): _description_
        res (int, optional): _description_. Defaults to 1.
        hfov_d (int, optional): _description_. Defaults to 360.
        v_max (_type_, optional): _description_. Defaults to np.pi/2.
        v_min (_type_, optional): _description_. Defaults to -np.pi/12.
        wrap (bool, optional): _description_. Defaults to False.
        blur (bool, optional): _description_. Defaults to False.
        blur_kernel_size (int, optional): _description_. Defaults to 3.

    Returns:
        2D Array: image 
    """
    X, Y, Z = world['X'], world['Y'], world['Z']
    dpi = 100
    hfov_deg = hfov_d
    hfov = np.deg2rad(hfov_deg)
    h_min = -hfov / 2
    h_max = hfov / 2

    vfov = v_max - v_min
    vfov_deg = np.rad2deg(vfov)

    resolution = res
    sky_colour = 'white'
    ground_colour = (0.1, 0.1, 0.1, 1)
    grass_colour = 'gray'
    grass_cmap = LinearSegmentedColormap.from_list('mycmap', [(0, (0, 0, 0, 1)), (1, grass_colour)])

    c = np.ones(Z.shape[0]) * 0.5

    image_ratio = vfov / hfov
    h_pixels = hfov_deg / resolution
    v_pixels = h_pixels * image_ratio

    im_width = h_pixels / dpi
    im_height = v_pixels / dpi

    fig = Figure(frameon=False, figsize=(im_width, im_height))
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.set_xlim(h_min, h_max)
    ax.set_ylim(v_min, v_max)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor(sky_colour)

    canvas = FigureCanvasAgg(fig)
    ground_verts = [[(h_min, v_min), (h_max, v_min), (h_max, 0), (h_min, 0)]]

    g = PolyCollection(ground_verts, facecolor=ground_colour, edgecolor='none')
    ax.add_collection(g)

    TH, PHI, R = cart2sph(X - x, Y - y, np.abs(Z) - z)
    TH_rel = pi2pi(TH - th)

    # fix the grass
    ind = (np.max(TH_rel, axis=1) - np.min(TH_rel, axis=1)) > np.pi
    TH_ext = np.vstack((TH_rel, np.mod(TH_rel[ind, :] - 2 * np.pi, -2 * np.pi)))
    n_blades = np.sum(ind)
    padded_ind = np.lib.pad(ind, (0, n_blades), 'constant')
    TH_ext[padded_ind, :] = np.mod(TH_rel[ind, :] + 2 * np.pi, 2 * np.pi)

    PHI_ext = np.vstack((PHI, PHI[ind, :]))
    R_ext = np.vstack((R, R[ind, :]))


    grass_verts = np.dstack((TH_ext, PHI_ext))
    p = PolyCollection(grass_verts, array=c, cmap=grass_cmap, edgecolors='none')
    ax.add_collection(p)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0, dpi=dpi)
    buf.seek(0)
    im = Image.open(buf)
    im_array = np.asarray(im)[:, :, 0:3]

    # grey scale and blurred image
    img_cv = cv2.cvtColor(np.asarray(im_array), cv2.COLOR_RGB2BGR)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    if wrap:
        img_cv = img_wrapper(img_cv, 1)
    if blur:
        img_cv = cv2.blur(img_cv, (blur_kernel_size, blur_kernel_size))
    return img_cv


def cuboid_3d_vertics(x_size, y_size, z_size, x, y, z):
    # a cuboid with (x_size,y_size,z_size) composed of triangles with the bottom-left edge located at (x,y,z)
    cuboid_p = [[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[1,0,0],[1,1,0]],
                [[0,0,0],[0,0,1],[0,1,1]],[[0,0,0],[0,1,0],[0,1,1]],
                [[0,0,0],[0,0,1],[1,0,1]],[[0,0,0],[1,0,0],[1,0,1]],
                [[0,0,1],[0,1,1],[1,1,1]],[[0,0,1],[1,0,1],[1,1,1]],
                [[1,0,0],[1,0,1],[1,1,1]],[[1,0,0],[1,1,0],[1,1,1]],
                [[0,1,0],[0,1,1],[1,1,1]],[[0,1,0],[1,1,0],[1,1,1]]]

    verts_xyz = np.zeros([3,len(cuboid_p),3])
    for i in range(len(cuboid_p)):
        zipped = list(zip(*cuboid_p[i]))
        verts_xyz[0,i] = np.array(zipped[0])
        verts_xyz[1,i] = np.array(zipped[1]) 
        verts_xyz[2,i] = np.array(zipped[2])

    verts_xyz[0] *= x_size
    verts_xyz[1] *= y_size
    verts_xyz[2] *= z_size
    
    verts_xyz[0] += x
    verts_xyz[1] += y
    verts_xyz[2] += z
    
    return verts_xyz


def construct_world(obstackles, foods, food_catchment,
                    start_pos, arena,
                    scale=1, background=None):
    # obstackles
    obs_vertex = np.zeros([3, 12*len(obstackles), 3])
    k = 0
    for i, (obs_k, obs_v) in enumerate(obstackles.items()):
        verts = cuboid_3d_vertics(obs_v['lengths'][0]*scale,
                                  obs_v['lengths'][1]*scale,
                                  obs_v['lengths'][2]*scale,
                                  obs_v['position'][0]*scale,
                                  obs_v['position'][1]*scale,
                                  obs_v['position'][2]*scale)
        obs_vertex[0, k:k+12] = verts[0]
        obs_vertex[1, k:k+12] = verts[1]
        obs_vertex[2, k:k+12] = verts[2]
        k += 12
    
    # background
    if background is not None:
        new_x = np.vstack([background['X']*scale,obs_vertex[0]])
        new_y = np.vstack([background['Y']*scale,obs_vertex[1]])
        new_z = np.vstack([background['Z']*scale,obs_vertex[2]])
        world = {'X': new_x,
                 'Y': new_y,
                 'Z': new_z,
                 'FoodSites': foods*scale,
                 'FoodCatchment': food_catchment*scale,
                 'Arena': arena*scale,
                 'StartPos': start_pos*scale
                 }
    else:
        world = {'X': obs_vertex[0],
                 'Y': obs_vertex[1],
                 'Z': obs_vertex[2],
                 'FoodSites': foods*scale,
                 'FoodCatchment': food_catchment*scale,
                 'Arena': arena*scale,
                 'StartPos': start_pos*scale
             }

    return world


def move_obstackles(obstackles, obstackles_num,
                    speed, T,
                    with_background=False):
    Xt = np.zeros([T, int(obstackles_num*12), 3])
    Yt = np.zeros([T, int(obstackles_num*12), 3])
    
    if with_background:
        init_x = obstackles['X'][-int(obstackles_num*12):]
        init_y = obstackles['Y'][-int(obstackles_num*12):]
        Z = obstackles['Z'][-int(obstackles_num*12):]
    else:
        init_x = obstackles['X']
        init_y = obstackles['Y']
        Z = obstackles['Z']

    Xt[0] = init_x
    Yt[0] = init_y

    for t in range(1, T):
        for n in range(obstackles_num):
            Xt[t, 12*n:12*(n+1)] = Xt[t-1, 12*n:12*(n+1)] + speed[t, n, 0]
            Yt[t, 12*n:12*(n+1)] = Yt[t-1, 12*n:12*(n+1)] + speed[t, n, 1]
    for n in range(obstackles_num):
        d = np.random.choice([-1, 1])
        if d == -1:
            Xt[:, 12*n:12*(n+1)] = Xt[:, 12*n:12*(n+1)][::-1]
            Yt[:, 12*n:12*(n+1)] = Yt[:, 12*n:12*(n+1)][::-1]
    
    return Xt, Yt, Z


def construct_dynamic_world(world, Xt, Yt, Z, period, obstackles_num):
    world_dic = {}
    for t in range(period):
        world_dic.update({str(t):{
            'X': np.vstack([world['X'][:-int(obstackles_num*12)], Xt[t]]),
            'Y': np.vstack([world['Y'][:-int(obstackles_num*12)], Yt[t]]),
            'Z': np.vstack([world['Z'][:-int(obstackles_num*12)], Z]),
        }})
    for k, v in world.items():
        if (k not in ('X','Y','Z')) and (not k.startswith('__')):
            world_dic.update({k:v})
    
    return world_dic


def get_collision_info(world, agent_pos, dynamic=False):
    """get the real collision information

    Args:
        world (dic): the world data containing obstacles position information
        agent_pos (array): the positions of the agent
        dynamic (bool): whether the obstacles are dynamic

    Return:
        collision (list): list of the time stamp of collision happened 
    """
    T = len(agent_pos)
    collision_t = []
    if dynamic:
        max_obs_t = max([int(s) for s in list(filter(lambda x:x.isnumeric(),
                                                     world.keys()))]) + 1
        obs_num = int(len(world['0'][0][0]['X'])/12)
        for t in range(T):
            if t // max_obs_t % 2 == 0:
                ind_t = str(t % max_obs_t)
            else:
                ind_t = str(max_obs_t - (t % max_obs_t) - 1)
            for n in range(obs_num):
                obs_pos = world[ind_t][0][0]
                obs_x = (obs_pos['X'][n*12][0] + obs_pos['X'][n*12][2])/2
                obs_y = (obs_pos['Y'][n*12][0] + obs_pos['Y'][n*12][2])/2
                obs_l_x = abs(obs_pos['X'][n*12][2]-obs_pos['X'][n*12][0])/2
                obs_l_y = abs(obs_pos['Y'][n*12][2]-obs_pos['Y'][n*12][0])/2
                if (abs(agent_pos[t][0] - obs_x) <= obs_l_x) and \
                    (abs(agent_pos[t][1] - obs_y) <= obs_l_y):
                    collision_t.append(t)
    else:
        obs_num = int(len(world['X'])/12)
        obs_x = np.zeros(obs_num)
        obs_y = np.zeros(obs_num)
        obs_l_x = np.zeros(obs_num)
        obs_l_y = np.zeros(obs_num)
        for n in range(obs_num):
            obs_x[n] = (world['X'][n*12][0] + world['X'][n*12][2])/2
            obs_y[n] = (world['Y'][n*12][0] + world['Y'][n*12][2])/2
            obs_l_x[n] = abs(world['X'][n*12][2]-world['X'][n*12][0])/2
            obs_l_y[n] = abs(world['Y'][n*12][2]-world['Y'][n*12][0])/2
        for t in range(T):
            for n in range(obs_num):
                if (abs(agent_pos[t][0] - obs_x[n]) <= obs_l_x[n]) and \
                    (abs(agent_pos[t][1] - obs_y[n]) <= obs_l_y[n]):
                        collision_t.append(t)
    # filter
    if len(collision_t) != 0:
        ind = [i+1 for i in np.where(np.diff(collision_t)>1)]
        ind = [0] + ind[0].tolist()
        collision_t = [collision_t[i] for i in ind]
    return collision_t


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    

def get_metrics(data, key):
    return [data[k][key] for k in data.keys()]


def get_route_length(pos):
    a = np.diff(pos, axis=0)
    return np.sum(np.sqrt(a[:, 0]**2 + a[:, 1]**2))


def get_trials_data(dynamic_world=False, mode='LGMD', key='headings'):
    results_dir = 'results/simulation_navigation'
    trial_data = []
    for obs_num in range(6, 48, 6):
        for n in range(15):
            try:
                if not dynamic_world:
                    filename = results_dir + '/{}Obs_T100/Static/{}/{}.mat'.format(
                        obs_num, mode, n)
                    data = sio.loadmat(filename)
                else:
                    filename = results_dir + '/{}Obs_T100/Dynamic/{}/{}.mat'.format(
                        obs_num, mode, n)
                    data = sio.loadmat(filename)
            except:
                print(filename + ' do not exit.')
            trial_data.append(data[key])
    return trial_data


def get_heatmap_data(dynamic_world=False, mode='LGMD'):
    results_dir = 'results/simulation_navigation'
    world = sio.loadmat('worlds/3FoodS200_Trans/6Obs_T100.mat')
    # stardarise
    xmin, xmax, ymin, ymax = world['Arena'][0]
    size = int((xmax - xmin))
    if not dynamic_world:
        heatmap_counter = np.zeros([size,size], dtype=np.uint8)
    else:
        heatmap_value = np.zeros([size,size])
        heatmap_counter = np.zeros([size,size], dtype=np.uint8)
    for obs_num in range(6, 48, 6):
        for n in range(15):
            try:
                if not dynamic_world:
                    filename = results_dir + '/{}Obs_T100/Static/{}/{}.mat'.format(
                        obs_num, mode, n)
                    data = sio.loadmat(filename)
                    pos = data['position']
                    for p in pos:
                        heatmap_counter[np.min([int(p[1]-ymin), size-1]),
                                        np.min([int(p[0]-xmin), size-1])]+=1
                else:
                    filename = results_dir + '/{}Obs_T100/Dynamic/{}/{}.mat'.format(
                        obs_num, mode, n)
                    data = sio.loadmat(filename)
                    pos = data['position']
                    v = np.sqrt(data['velocities'][:,0]**2+data['velocities'][:,1]**2)
                    for i, p in enumerate(pos):
                        heatmap_value[np.min([int(p[1]-ymin), size-1]),
                                        np.min([int(p[0]-xmin), size-1])]+=v[i]
                        heatmap_counter[np.min([int(p[1]-ymin), size-1]),
                                        np.min([int(p[0]-xmin), size-1])]+=1
            except:
                print(filename + ' do not exit.')
    if dynamic_world:
        return heatmap_value/(heatmap_counter+1)
    else:
        return heatmap_counter