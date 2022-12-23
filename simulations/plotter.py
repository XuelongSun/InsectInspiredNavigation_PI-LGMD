from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mpl_toolkits.mplot3d.art3d as art3d 
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import matplotlib.collections as mcoll

plt.rcParams['font.size'] = 16


def plot_grid_figure(row, column, axes_span, figsize=(12, 12), projection_3d=None):
    """[generated a figure based on grid (row x column)]

    Args:
        row ([type]): [description]
        column ([type]): [description]
        axes_span (list, positional): [[row_s, row_e, column_s, column_e],
                                      [row_s, row_e, column_s ,column_e],...]
        figsize (tuple, optional): [description]. Defaults to (12, 12).

    Returns:
        [type]: [description]
    """
    fig = plt.figure(figsize=figsize)
    axes = []
    gs = GridSpec(row, column)
    for i, p in enumerate(axes_span):
        if projection_3d:
            if i in projection_3d:
                axes.append(plt.subplot(gs[p[0]:p[1], p[2]:p[3]], projection='3d'))
            else:
                axes.append(plt.subplot(gs[p[0]:p[1], p[2]:p[3]]))
        else:
            axes.append(plt.subplot(gs[p[0]:p[1], p[2]:p[3]]))

    return fig, axes


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline(x, y, z=None, 
              cmap=plt.get_cmap('copper'),
              norm=plt.Normalize(0.0, 1.0),
              linewidth=2, alpha=1.0):
    """
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.2, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    return lc


def plot_food_sites(fig, ax, food_sites, food_catchment):
    """_summary_

    Args:
        food_sites (_type_): _description_
        food_catchment (_type_): _description_
    """
    for i, f in enumerate(food_sites):
        ax.add_patch(
            patches.Circle(
                xy=(f[0], f[1]),
                radius=food_catchment[i],
                fc='lightblue',    # facecolor
                ec='cornflowerblue',
                alpha=0.3
            )
        )
        ax.text(f[0]-food_catchment[i], 
                f[1], 
                'FoodSite{}'.format(i+1))

    return fig, ax


def plot_obstacle(fig, ax, obs_x, obs_y, obs_z=None):
    if obs_z is None:
        obj_verts = np.dstack([obs_x, obs_y])
        objs = PolyCollection(obj_verts, 
                              facecolor='k', edgecolor='none',
                              alpha=1)
        ax.add_collection(objs)
    else:
        obj_verts = np.dstack([obs_x, obs_y, obs_z])
        objs = art3d.Poly3DCollection(obj_verts,
                                facecolor='k', edgecolor='none',
                                alpha=1)
        ax.add_collection(objs)
    
    return fig, ax


def plot_agent(fig, ax, pos, show_trajectory=False):
    current_pos = pos[-1]
    if show_trajectory:
        ax.scatter(current_pos[0], current_pos[1],
                   marker='o', color='skyblue', s=10)
        ax.plot(pos[:, 0], pos[:, 1],
                ls='--', lw=0.5, color='skyblue')
    else:
        ax.scatter(current_pos[0], current_pos[1],
                   marker='o', color='skyblue', s=10)
    
    return fig, ax


def plot_arena(fig, ax, world):
    # arena boundaries
    arena_boundary = world['Arena'][0]
    margin = 4
    rect = patches.Rectangle([arena_boundary[0]-2, arena_boundary[2]-2],
                             arena_boundary[1] - arena_boundary[0]+2,
                             arena_boundary[3] - arena_boundary[2]+2,
                             fill=False, lw=6, ls='-', ec='gray')
    ax.add_patch(rect)
    ax.set_xlim(arena_boundary[0] - margin,
                arena_boundary[1] + margin)
    ax.set_ylim(arena_boundary[2] - margin,
                arena_boundary[3] + margin)
    # start point
    sp = world['StartPos'][0]
    ax.scatter(sp[0], sp[1], marker='*', color="red",
               s=50)
    ax.text(sp[0]+2, sp[1], 'Start Point', fontsize=14, color='k')
    
    # food
    fig, ax = plot_food_sites(fig, ax,
                              world['FoodSites'],
                              world['FoodCatchment'][0])
    return fig, ax, margin


def plot_single_state(world, data, t,
                      arena_only=True,
                      dynamic=False,
                      in_3d=False):
    """_summary_

    Args:
        data (dic): _description_
        t (int): _description_
        arena_only (bool, optional): only show info in arena. Defaults to True.
    """
    max_obs_t = max([int(s) for s in list(filter(lambda x:x.isnumeric(),
                                                 world.keys()))]) + 1
    if arena_only:
        if in_3d:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_axes(projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_axis_off()
            
            # plot arena
            fig, ax, margin = plot_arena(fig, ax, world)
            
            # rulers
            xlim = ax.get_xlim()
            xl = xlim[1]-xlim[0]
            ylim = ax.get_ylim()
            yl = ylim[1]-ylim[0]
            ruler_pos = [[xlim[1]-margin-xl*0.14,
                          xlim[1]-margin-xl*0.14+50],
                         [ylim[1]-margin-yl*0.06,
                          ylim[1]-margin-yl*0.06]]
            ax.plot(ruler_pos[0], ruler_pos[1], lw=4, color='black')
            ax.text(ruler_pos[0][0], ruler_pos[1][0]+6, '50 Steps',
                    fontsize=14)

            # plot obstacles
            t_s = t - 24
            t_e = t + 24
            if dynamic:
                for i in range(t_s, t_e, 6):
                    obs_x = world[str(i % max_obs_t)][0][0]['X']
                    obs_y = world[str(i % max_obs_t)][0][0]['Y']
                    obj_verts = np.dstack([obs_x, obs_y])
                    objs = PolyCollection(obj_verts,
                                          facecolor='k', edgecolor='w',
                                          alpha=0.01 + (i-t_s)/(t_e - t_s))
                    ax.add_collection(objs)
            else:
                obs_pos = world['0'][0][0]
                fig, ax = plot_obstacle(fig, ax, obs_pos['X'], obs_pos['Y'])
            
            # plot agent
            ax.scatter(data['position'][t_s:t_e:2, 0],
                       data['position'][t_s:t_e:2, 1],
                       marker='o',
                       color=['lightblue']*int((t_e-t_s)/2),
                       edgecolor='b',
                       label='agent',
                       alpha=[0.01 + (i-t_s)/(t_e - t_s)*0.9 for i in range(t_s, t_e, 2)])
            # highlight stop point
            ax.scatter(data['position'][t+1, 0],
                       data['position'][t+1, 1],
                       color='red',
                       marker='x',
                       s=30)
            ax.legend(loc=0, scatterpoints=1)
            # fig, ax = plot_agent(fig, ax, data['position'][:t],
            #                      show_trajectory=True)
    else:
        pass
    
    return fig, ax


def plot_3d_arena(world):
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    arena_boundary = world['Arena'][0]
    
    # obstacles
    ax.text(-100, 200, 0,
            'Obstacles',
            fontsize=12,)
    obj_verts = np.dstack([world['0'][0][0]['X'],
                           world['0'][0][0]['Y'],
                           world['0'][0][0]['Z']])
    obstackles = art3d.Poly3DCollection(obj_verts, facecolor='k',
                                        edgecolor='none', alpha=0.8)
    ax.add_collection(obstackles)
    # food sites
    food_sites = world['FoodSites']
    food_catchment = world['FoodCatchment'][0]
    for i, f in enumerate(food_sites):
        ax.text(f[0], f[1], 0, 'FoodSite' + str(i+1), fontsize=12)
        f_circle = patches.Circle(
                xy=(f[0], f[1]),
                radius=food_catchment[i],
                fc='lightblue',    # facecolor
                ec='cornflowerblue',
                alpha=0.3
            )
        ax.add_patch(f_circle)
        art3d.pathpatch_2d_to_3d(f_circle, z=0, zdir="z")
    # boundary
    rect = patches.Rectangle([arena_boundary[0], arena_boundary[2]],
                             arena_boundary[1] - arena_boundary[0],
                             arena_boundary[3] - arena_boundary[2],
                             fill=False, lw=4, ls='-', ec='gray')
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")
    # start point
    sp = world['StartPos'][0]
    ax.scatter(sp[0], sp[1], 0, color='r', marker='*', s=20)
    ax.text(sp[0], 0, 0, 'Start Point', fontsize=12)
    ax.set_xlim3d([arena_boundary[0], arena_boundary[1]])
    ax.set_ylim3d([arena_boundary[2], arena_boundary[3]])
    ax.set_zlim3d([0, arena_boundary[1]-arena_boundary[0]])
    # agent
    # ax.scatter(-100, -100, 0, marker='o',
    #            color='cyan', s=25)
    # ax.text(-105, -95, 0, 'Agent', fontsize=12)
    ax.set_xlabel('X / steps')
    ax.set_ylabel('Y / steps')
    ax.set_zlabel('Z')
    ax.grid(False)
    # background color
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    return fig, ax


def plot_whole_results(world, data, phase='exploration', dynamic=False,
                       st=0, et=0,
                       xclip=None, yclip=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    # trajectories data
    pos = data['position']
    # world data
    # arena
    fig, ax, margin = plot_arena(fig, ax, world)
    ax.set_axis_off()
    # rulers
    xlim = ax.get_xlim()
    xl = xlim[1]-xlim[0]
    ylim = ax.get_ylim()
    yl = ylim[1]-ylim[0]
    ruler_pos = [[xlim[1]-margin-xl*0.14,
                  xlim[1]-margin-xl*0.14+50],
                 [ylim[1]-margin-yl*0.06,
                  ylim[1]-margin-yl*0.06]]
    ax.plot(ruler_pos[0], ruler_pos[1], lw=4, color='black')
    ax.text(ruler_pos[0][0], ruler_pos[1][0]+6, '50 Steps',
            fontsize=14)
    
    if phase == 'exploration':
        # trajectories
        for i, t in enumerate(data['found_food_t'][0]):
            # exploring
            s_t = data['got_home_t'][0][i-1] if i>0 else 0
            ax.plot(pos[s_t:t, 0], pos[s_t:t, 1],
                    color='orange', lw=i/2+1, alpha=0.7,
                    label='exploring' if i==2 else '')
            # homing
            e_t = data['got_home_t'][0][i] if i < len(data['got_home_t'][0]) else \
                len(pos)
            ax.plot(pos[t:e_t, 0],
                    pos[t:e_t, 1],
                    color='green', lw=i/2+1, alpha=0.7,
                    label='homing' if i==2 else '')
        if e_t < len(pos):
            ax.plot(pos[e_t:, 0], pos[e_t:, 1],
                    color='orange', lw=4, alpha=0.7)
        
        ax.legend(numpoints=1)
        
        # vector memories
        food_sites = world['FoodSites']
        vector_memories = data['vector_memories']
        axes = [[]]*len(vector_memories)
        scale = abs(food_sites[0][1])
        for i, m in enumerate(vector_memories):
            axes[i] = ax.inset_axes([food_sites[i][0]-scale*0.25,
                                     food_sites[i][1]+scale*0.16,
                                     scale/2,
                                     scale/6], transform=ax.transData)
            
            axes[i].bar(range(1, len(m)+1), m,
                        color='orange', alpha=0.5)
            axes[i].set_title('Vector Memory (CPU4)', fontsize=14)
            axes[i].set_axis_off()
            axes[i].patch.set_alpha(0.0)
    elif phase == 'foraging':
        # time-bar
        colorbar_ax = ax.inset_axes([0.05, 0.92, 0.15, 0.03])
        colorbar_ax.text(0.1, -1.4, 'Time', fontsize=14)
        colorbar_ax.arrow(0.5, -1.1, 0.2, 0.0,
                          head_length=0.1, width=0.2, fc='k')
        fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(0, 1),
                                       cmap='Oranges'),
                     ax=ax, cax=colorbar_ax,
                     orientation='horizontal',)
        colorbar_ax.set_axis_off()
        colorbar_ax.set_ylim(-1.5, 1)
        # LGMD spiking
        for c_t in np.where(data['collision'][0]==1):
            ax.scatter(pos[c_t, 0], pos[c_t, 1], marker='x',
                       color='magenta', s=30,
                       label='LGMD Spiking')
            
        lg = ax.legend(loc=5, scatterpoints=1,
                       fontsize=12)
        lg.get_frame().set_alpha(0.0)
        
        # trajectories
        lc = colorline(pos[:, 0], pos[:, 1],
                       cmap='Oranges',linewidth=2.5)
        ax.add_collection(lc)
        lc.set_zorder(0)
        # obstacles if static
        if not dynamic:
            fig, ax = plot_obstacle(fig, ax,
                                    world['0'][0][0]['X'],
                                    world['0'][0][0]['Y'])
        else:
            max_obs_t = max([int(s) for s in list(filter(lambda x:x.isnumeric(),
                                                         world.keys()))]) + 1
            for i in range(0, max_obs_t, 6):
                obs_x = world[str(i % max_obs_t)][0][0]['X']
                obs_y = world[str(i % max_obs_t)][0][0]['Y']
                obj_verts = np.dstack([obs_x, obs_y])
                objs = PolyCollection(obj_verts,
                                      facecolor='k',
                                      edgecolor='none',
                                      alpha=0.04)
                ax.add_collection(objs)
        if st != 0:
            # add sample point
            ax.scatter(pos[st, 0], pos[st, 1], marker='^',
                       color='royalblue')
            # ax.text()
            ax.scatter(pos[et, 0], pos[et, 1], marker='^',
                       color='royalblue')
        
        if xclip is not None:
            rect = patches.Rectangle([xclip[0], yclip[0]],
                                     xclip[1] - xclip[0],
                                     yclip[1] - yclip[0],
                                     fill=False, lw=2,
                                     ls='--', ec='gray')
            ax.add_patch(rect)
            
    else:
        pass
    return fig, ax


def plot_dynamic_collision(world, data, st, et, xlim, ylim):
    xl = xlim[1] - xlim[0]
    yl = ylim[1] - ylim[0]
    maxl = np.max([xl, yl])
    interval = 12
    if et - st < 4 * interval:
        print('time period should bigger than {}, got {}'.format(interval*4,
                                                                et-st) )
        return None, None
    num = int(np.floor(np.sqrt((et - st) // interval)))
    fig, axes = plt.subplots(num, num, figsize=(6*xl/maxl, 6*yl/maxl))
    max_obs_t = max([int(s) for s in list(filter(lambda x:x.isnumeric(),
                                                 world.keys()))]) + 1
    for n, ax in enumerate(axes.flatten()):
        t_s = st + interval*((n // num)*num + (n % num))
        t_e = t_s + interval
        for i in range(t_s, t_e, int(interval/4)):
            obs_x = world[str(i % max_obs_t)][0][0]['X']
            obs_y = world[str(i % max_obs_t)][0][0]['Y']
            obj_verts = np.dstack([obs_x, obs_y])
            objs = PolyCollection(obj_verts,
                                  facecolor='k', edgecolor='none',
                                  alpha=0.1 + (i - t_s)/(t_e - t_s)*0.9)
            ax.add_collection(objs)
        ax.scatter(data['position'][t_s:t_e:2, 0],
                   data['position'][t_s:t_e:2, 1],
                   marker='o',
                   color=['orange']*int((t_e-t_s)/2),
                   edgecolor='orange',
                   label='agent',
                   alpha=[0.1 + (i-t_s)/(t_e - t_s)*0.9 for i in range(t_s, t_e, 2)])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_axis_off()
        rect = patches.Rectangle([xlim[0], ylim[0]],
                                 xlim[1] - xlim[0],
                                 ylim[1] - ylim[0],
                                 fill=False, lw=6, ls='-', ec='gray')
        ax.add_patch(rect)
        ax.set_aspect(1)
        ax.set_title('t = {}~{}'.format(t_s, t_e), fontsize=14)
        ax.grid()
    return fig, axes


def plot_trials_result(data1, data2,
                       colors=['salmon', 'skyblue'],
                       labels=['data1', 'data2'],
                       title='Title',
                       xlabel='x',
                       ylabel='y',
                       legend_loc=0,
                       base_y=None,
                       showMedianLine=True,
                       showReduceRate=False,
                       showVariance=False,
                       showNum=False,
                       showBaseLine=False):
    fig, ax = plt.subplots(figsize=(16, 9))
    xp = np.array(range(len(data1)))
    xp1 = xp*2-0.3
    xp2 = xp*2+0.3
    v1 = plt.violinplot(data1,
                        positions=xp1,
                        showmedians=False)
    v2 = plt.violinplot(data2,
                        positions=xp2,
                        showmedians=False)
    
    for p in ['cmaxes', 'cmins', 'cbars']:
        v1[p].set_color(colors[0])
        v1[p].set_linewidth(3)
        v2[p].set_linewidth(3)
        v2[p].set_color(colors[1])
    
    for pc1 in v1['bodies']:
        pc1.set_edgecolor(colors[0])
        pc1.set_facecolor(colors[0])
        pc1.set_alpha(0.4)
    
    for pc2 in v2['bodies']:
        pc2.set_edgecolor(colors[1])
        pc2.set_facecolor(colors[1])
        pc2.set_alpha(0.4)
    
    if showMedianLine:
        ax.plot(xp*2, [np.median(x) for x in data1],
                color=colors[0], lw=3, ls='--')
        ax.plot(xp*2, [np.median(x) for x in data2],
                color=colors[1], lw=3, ls='--')
        ax.scatter(xp*2, [np.median(x) for x in data1],
                   color=colors[0], marker='d', s=100,
                   edgecolor='gray')
        ax.scatter(xp*2, [np.median(x) for x in data2],
                   color=colors[1], marker='d', s=100,
                   edgecolor='gray')
        ax.scatter([],[],marker='d',s=100, color='white',
                   edgecolor='gray',
                   label='Medians')
        
    ax.set_xticklabels([str(6*x) for x in range(len(data1)+1)])
    data1_max= [np.max(t) for t in data1]
    data2_max= [np.max(t) for t in data2]
    
    data1_min= [np.min(t) for t in data1]
    data2_min= [np.min(t) for t in data2]
    # ax.set_yticks(range(0,
    #                     int(np.max([np.max(data1_max), np.max(data2_max)]))+5,
    #                     5))
    if base_y is not None:
        xmin = np.min([np.min(data1_min), np.min(data2_min)])
        ax.set_ylim(np.min([base_y, xmin])*0.9,
                    np.max([np.max(data1_max), np.max(data2_max)])*1.1)
    else:
        ax.set_ylim(np.min([np.min(data1_min), np.min(data2_min)])*0.9,
                    np.max([np.max(data1_max), np.max(data2_max)])*1.1)
    # dummy plot for legend
    ax.plot([], [], color=colors[0], lw=3, label=labels[0])
    ax.plot([], [], color=colors[1], lw=3, label=labels[1])
    
    
    for i in range(len(data1)):
        # reduce rate
        if showReduceRate:
            if np.median(data1[i]) != 0:
                rate = (np.median(data1[i]) - np.median(data2[i]))/np.median(data1[i])
                show_rate = '{:.2f}%'.format(rate*100)
            else:
            # rate = (np.sum(data1[i]) - np.sum(data2[i]))/np.sum(data1[i])
            # rate = (np.max(data1[i]) - np.min(data2[i]))/np.max(data1[i])
                show_rate = 'No Collision'
            ax.annotate(show_rate,
                        ((xp*2)[i], np.median(data2[i])),
                        ((xp*2)[i], np.median(data1[i])),
                        arrowprops={'width': 1, 'headwidth':5,
                                    'facecolor': 'gray'}
                        )
        # variance
        if showVariance:
            ax.text(xp1[i]-0.6, np.max(data1[i])+0.5,
                    r'$\sigma={:.2f}$'.format(np.var(data1[i])),
                    color=colors[0], fontsize=16, fontweight='bold')
            ax.text(xp2[i]-0.6, np.min(data2[i])-1.5,
                    r'$\sigma={:.2f}$'.format(np.var(data2[i])),
                    color=colors[1], fontsize=16, fontweight='bold')
        
        if showNum:
            ax.text(xp1[i]-0.6, np.max(data1[i])+2,
                    r'$N={:d}$'.format(len(data1[i])),
                    color=colors[0], fontsize=16)
            ax.text(xp2[i]-0.6, np.min(data2[i])-2.7,
                    r'$N={:d}$'.format(len(data2[i])),
                    color=colors[1], fontsize=16)
    ax.set_xlim(-1, len(data1)*2-1)
    if showBaseLine:
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [base_y, base_y],
                color='gray', lw=3, ls='-.', label='Optimal')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc)
    ax.grid()
    ax.set_title(title)
    return fig, ax
    
    
def plot_trials_result_compare(data,
                               title='Title',
                               subtitles='1234',
                               colors=['salmon', 'skyblue'],
                               labels=['d1','d2'],
                               baseline=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    vios_1 = []
    vios_2 = []
    xp = np.array(range(len(data[0,0])))
    xp1 = xp*2-0.3
    xp2 = xp*2+0.3
    if baseline is not None:
        xmin_left = baseline[0]-abs(baseline[0])*0.1
        xmin_right = baseline[1]-abs(baseline[1])*0.1
    else:
        xmin_left = np.min([data[0].min(), data[2].min()])
        xmin_right = np.min([data[1].min(), data[3].min()])
    
    xmax_left = np.max([data[0].max(), data[2].max()])*1.02
    xmax_right = np.max([data[1].max(), data[3].max()])*1.02
    
    for i, ax in enumerate(axes.flatten()):
        vios_1.append(ax.violinplot(data[i, 0].T,
                                  positions=xp1,
                                  showmedians=True,
                                  vert=False))
        vios_2.append(ax.violinplot(data[i, 1].T,
                            positions=xp2,
                            showmedians=True,
                            vert=False))
        if i%2 == 0:
            ax.plot([baseline[0], baseline[0]], [xp[0]-1, xp[-1]*4],
                    color='gray', lw=3, ls='--', label='Optimal')
            ax.set_xlim(xmin_left, xmax_left)
            ax.set_ylabel('Number of Obstacles')
        else:
            ax.plot([baseline[1], baseline[1]], [xp[0]-1, xp[-1]*4],
                    color='gray', lw=3, ls='--', label='Optimal')
            ax.set_xlim(xmin_right, xmax_right)
        
        if i == 2:
            ax.set_xlabel('Route Length / Steps')
        elif i == 3:
            ax.set_xlabel('Finish Time / Time-steps')
        else:
            # # ax.spines['bottom'].set_visible(False)
            # ax.set_xticklabels([])
            # ax.set_xticks([])
            # ax.get_xaxis().set_visible(False)
            pass
        
        for p in ['cmaxes', 'cmins', 'cbars', 'cmedians']:
            vios_1[i][p].set_color(colors[0])
            vios_1[i][p].set_linewidth(1.5)
            vios_2[i][p].set_linewidth(1.5)
            vios_2[i][p].set_color(colors[1])
        
        for pc1 in vios_1[i]['bodies']:
            pc1.set_edgecolor(colors[0])
            pc1.set_facecolor(colors[0])
            pc1.set_alpha(0.4)
        
        for pc2 in vios_2[i]['bodies']:
            pc2.set_edgecolor(colors[1])
            pc2.set_facecolor(colors[1])
            pc2.set_alpha(0.4)

        ax.set_yticks(xp*2)
        ax.set_yticklabels([str(6*x) for x in range(1, len(xp)+1)])
        ax.set_title(subtitles[i], fontsize=14)
        ax.set_ylim(xp1[0]-1, xp2[-1]+1)
        ax.set_xticks(range(0,
                            int(ax.get_xlim()[1]/1000)*1100,
                            1000))
        ax.grid(axis='x', ls=':', lw=0.8, color='k', alpha=0.5)
        
            # dummy plot for legend
        ax.plot([], [], color=colors[0], lw=2, label=labels[0])
        ax.plot([], [], color=colors[1], lw=2, label=labels[1])
        
        ax.legend(loc=4, fontsize=12)
    
    return fig, axes

def plot_data_heatmap(data, world, cmap='PuRd',
                      label='label',
                      text1='t1', text2='t2'):
    fig, ax = plt.subplots(figsize=(10,10))
    # fig, ax, m = plot_arena(fig, ax, world)
    xmin, xmax, ymin, ymax = world['Arena'][0]
    size = int((xmax - xmin))
    x,y = np.meshgrid(np.linspace(xmin, xmax, size), 
                      np.linspace(ymin, ymax, size))
    pc = ax.pcolormesh(x, y, data*10,
                       shading='auto', cmap=cmap)
    colorbar_ax = ax.inset_axes([0.05, 0.92, 0.15, 0.03])
    vmin = np.min(data)
    vmax = np.max(data)
    fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin, vmax),
                                   cmap=cmap),
                 ax=ax, cax=colorbar_ax,
                 orientation='horizontal',
                 label=label)
    ax.set_title(text2)
    ax.text(110, 220, text1,
            color='orange', fontsize=16, fontweight='bold')
    # ax.text(40, 200, text2,
    #     color='orange', fontsize=14)
    ax.set_aspect(1)
    ax.set_axis_off()
    
    return fig, ax

def plot_data_heatmap2(data, 
                       world, cmap='PuRd',
                       text_color='purple',
                      label='label',
                      text='t', titles=('t1', 't2')):
    fig, axes = plt.subplots(1,2,figsize=(16, 8))
    xmin, xmax, ymin, ymax = world['Arena'][0]
    size = int((xmax - xmin))
    x,y = np.meshgrid(np.linspace(xmin, xmax, size), 
                      np.linspace(ymin, ymax, size))
    vmin = np.min(data)
    vmax = np.max(data)
    
    for i, ax in enumerate(axes):
        ax.pcolormesh(x, y, (data[i]*10)/vmax*np.max(data[i]),
                      shading='auto', cmap=cmap)
        colorbar_ax = ax.inset_axes([0.1, 0.92, 0.15, 0.03])

        plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin, vmax),
                                    cmap=cmap),
                    ax=ax, cax=colorbar_ax,
                    orientation='horizontal',
                    label=label)
        ax.set_title(titles[i])
        ax.text(110, 220, text,
                color=text_color, fontsize=16, fontweight='bold')
        # ax.text(40, 200, text2,
        #     color='orange', fontsize=14)
        ax.set_aspect(1)
        ax.set_axis_off()
    
    return fig, axes