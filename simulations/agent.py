import numpy as np
import copy

from models import CXRate, LobulaGiantMotionDetectorModel
from useful_tools import get_img_view
from useful_tools import get_next_state


class NavigationAgent:
    def __init__(self, noise=0.1):
        self.cx = CXRate()
        self.cx.noise = noise
        
        self.lgmd = LobulaGiantMotionDetectorModel(200, 200)
        self.hfov_d = 120
        
        self.reset()
        
        # LGMD parameter - static
        self.lgmd.ffi_thr = 0.2
        self.lgmd.i_w = 0.35
        
        self.collision_turn_flag = 0
    
    def update_cells(self, heading, velocity, tb1, memory, filtered_steps=0.0):
        """Generate activity for all cells, based on previous activity and current
        motion."""
        # Compass
        tl2 = self.cx.tl2_output(heading)
        cl1 = self.cx.cl1_output(tl2)
        tb1 = self.cx.tb1_output(cl1, tb1)

        # Speed
        flow = self.cx.get_flow(heading, velocity, filtered_steps)
        tn1 = self.cx.tn1_output(flow)
        tn2 = self.cx.tn2_output(flow)

        # Update memory for distance just travelled
        memory = self.cx.cpu4_update(memory, tb1, tn1, tn2)
        cpu4 = self.cx.cpu4_output(memory)

        # Steer based on memory and direction
        cpu1 = self.cx.cpu1_output(tb1, cpu4)
        motor = self.cx.motor_output(cpu1)
        return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor

    def reset(self):
        self.current_state = 0
        
        self.food_found_index = []
        self.step_found_food = []
        self.step_got_home = []
        self.pi_memory_snapshot = 0.5 * np.ones(16)
        self.vector_memories = []
        self.lgmd.spikes = []
        self.foraging_initialized = False
        self.current_forage_food_index = None
        self.foraged_food_index = []
        self.foraged_food_time = []
        self.forage_completed = False
        
        self.near_goal_pos = []
        
        self.has_explored = False
        
        self.out_arena = False
        
        self.boundary_counter = 0
        self.boundary_margin = 0
        self.collision_counter = 0
        
    def set_foraging_initialized(self, vector_memories=None, 
                                 food_found_index=None):
        self.current_state = 2
        self.foraged_food_index = []
        self.foraging_initialized = False
        self.lgmd.spikes = []
        self.foraged_food_time = []
        self.forage_completed = False
        
        self.near_goal_pos = []
        self.has_explored = True
        self.out_arena = False
        self.boundary_counter = 0
        self.collision_counter = 0
        if vector_memories is not None:
            self.vector_memories = vector_memories
        if food_found_index is not None:
            self.food_found_index = food_found_index
        
        self.boundary_margin = 0

    def run(self, world, start_h, time_out,
            dynamic_world=False,
            collision_detection=True,
            log=False, c_decay=0):

        pos = [[world['StartPos'][0][0],
                world['StartPos'][0][1]]]
        h = [start_h]

        velocity = [[0., 0.]]
        
        states = [self.current_state]

        food_sites = world['FoodSites']
        food_radius = world['FoodCatchment'][0]
        boundary = world['Arena'][0]

        # PI home vector memory
        pi_memory = [np.zeros([16])]

        # LGMD model output
        lgmd_mp = [0]
        ffi_mp = [0]

        if dynamic_world:
            dyn_max_t = max([int(s) for s
                             in list(filter(lambda x:x.isnumeric(),
                                            world.keys()))]) + 1
            img_t = []
            # back and forth
            for t in range(time_out):
                if t // dyn_max_t % 2 == 0:
                    img_t.append(str(t % dyn_max_t))
                else:
                    img_t.append(str(dyn_max_t - (t % dyn_max_t) - 1))
            self.lgmd.lgmd_thr = 0.96 # 0.96
            self.lgmd.spike_interval = 8 # 8 #5
        else:
            img_t = ['0']*time_out
            self.lgmd.lgmd_thr = 0.98
            self.lgmd.spike_interval = 7

        img_pre = get_img_view(world[img_t[0]][0][0],
                               pos[0][0], pos[0][1],
                               0.01, h[0], hfov_d=self.hfov_d,
                               wrap=False, blur=False)
        # alter the contrast
        img_pre[np.where(img_pre >= c_decay)] -= c_decay
        self.lgmd.img_height = img_pre.shape[0]
        self.lgmd.img_width = img_pre.shape[1]
        self.lgmd.reset()
        self.lgmd.frame[0] = img_pre

        # Initialise TB and memory
        tb1 = np.zeros(8)
        memory = 0.5 * np.ones(16)

        for t in range(time_out):
            img = get_img_view(world[img_t[t]][0][0],
                               pos[t][0], pos[t][1],
                               0.01, h[t], hfov_d=self.hfov_d,
                               wrap=False, blur=False)
            # low contrast testing
            img[np.where(img >= c_decay)] -= c_decay
            self.lgmd.frame[1] = copy.deepcopy(img)
            # p, i, s, Ce, g, lgmd, ffi = self.lgmd.run(luminance_per_num=1,
            #                                 fixed_ts=True,
            #                                 fixed_ffi_thr=True)
            p, i, s, Ce, g, lgmd, ffi = self.lgmd.run_no_enhance(luminance_per_num=1,
                                                      fixed_ts=True,
                                                      fixed_ffi_thr=True)
            self.lgmd.frame[0] = copy.deepcopy(img)
            lgmd_mp.append(lgmd)
            ffi_mp.append(ffi)
            # CX path integration
            tl2, cl1, tb1, tn1, tn2, memory, \
                cpu4, cpu1, pi_motor = self.update_cells(
                    heading=h[t],
                    velocity=np.array(velocity)[t],
                    tb1=tb1,
                    memory=memory)
            pi_memory.append(memory.copy())
            # inner state driven desired moving direction
            if self.current_state == 0:
                # explore mode - randome walking
                motor = np.random.vonmises(0.0, 100.0, 1)[0]*0.5
            elif (self.current_state == 3) or (self.current_state == 4):
                # near-range sensor driving
                goal_dir = np.arctan2(self.near_goal_pos[0]-pos[t][0],
                                      self.near_goal_pos[1]-pos[t][1])
                motor = goal_dir - h[t]
            else:
                # homing using vector navigation
                motor = pi_motor*1.0
            
            if (self.boundary_counter == 0) and (self.collision_counter == 0):
                # if at the boundary
                if (pos[t][0] < boundary[0]+self.boundary_margin) \
                    or (pos[t][0] > boundary[1]-self.boundary_margin) \
                    or (pos[t][1] < boundary[2]+self.boundary_margin) \
                    or (pos[t][1] > boundary[3]-self.boundary_margin):
                        h_tmp = h[t].copy()
                        v_tmp = [0, 0]
                        self.boundary_counter = 5
                else:
                    if (self.lgmd.spikes[-1] == 1) and (collision_detection):
                        if dynamic_world:
                            # 1.stop
                            h_tmp = h[t].copy()
                            v_tmp = [0, 0]
                            self.collision_counter = 2
                        else:
                            # 2.random turn left or right
                            v_m = np.sqrt(np.sum(velocity[t][0]**2+velocity[t][1]**2))
                            h_tmp, v_tmp = get_next_state(
                                    h[t], np.array(velocity[t]),
                                    np.pi/2*np.random.choice([-1, 1]),
                                    acceleration=0.3*v_m, drag=0.1)
                            self.collision_counter = 2
                    else:
                        h_tmp, v_tmp = get_next_state(
                            h[t], np.array(velocity[t]), motor,
                            acceleration=0.4, drag=0.2)
            else:
                h_tmp = h[t].copy()
                v_tmp = velocity[t].copy()
                if self.boundary_counter > 0:
                    if self.boundary_counter >= 5:
                        h_tmp, v_tmp = get_next_state(
                                    h[t], np.array(velocity[t]),
                                    3*np.pi/4,
                                    acceleration=0.4, drag=0.1)
                    self.boundary_counter -= 1
                if self.collision_counter > 0:
                    self.collision_counter -= 1

            h.append(h_tmp)
            velocity.append([v_tmp[0], v_tmp[1]])
            pos.append([pos[t][0] + v_tmp[0],
                        pos[t][1] + v_tmp[1]])

            # state calculation
            if self.current_state == 0:
                # exploring
                for i, f in enumerate(food_sites):
                    if i not in self.food_found_index:
                        if np.sqrt((pos[t+1][0] - f[0])**2 +
                                   (pos[t+1][1] - f[1])**2) <= food_radius[i]:
                            # near reward
                            if log:
                                print('''***t={}:near food {}.'''.format(t, i+1))
                            self.current_state = 3
                            self.food_found_index.append(i)
                            self.near_goal_pos = f.copy()
                            break

            elif self.current_state == 1:
                # homing
                if np.sqrt((pos[t+1][0] - pos[0][0])**2 +
                           (pos[t+1][1] - pos[1][1])**2) <= food_radius[0]:
                    # near home
                    self.current_state = 4
                    self.near_goal_pos = pos[0].copy()
                    
            elif self.current_state == 2:
                # foraging
                if self.foraging_initialized:
                    ind = self.food_found_index[self.current_forage_food_index]
                    dis = np.sqrt((pos[t+1][0] - food_sites[ind][0])**2 +
                                  (pos[t+1][1] - food_sites[ind][1])**2)
                    if dis <= 1:
                        if log:
                            print('''***t={}:got food {} via vector memory foraging'''.format(t,
                                                                                          ind + 1))
                        self.foraged_food_index.append(
                            self.current_forage_food_index)
                        self.foraged_food_time.append(t)
                        if len(self.foraged_food_index) == len(food_sites):
                            self.current_state = 1
                            if log:
                                p_s = '***t={}:got last food, homing...'
                                print(p_s.format(t))
                            memory = self.vector_memories[
                                self.current_forage_food_index].copy()
                            self.cx.cpu4_mem_gain = 0.005
                        else:
                            # got food, get next food site as target
                            diff = 100
                            min_ind = None
                            for i, f_m in enumerate(self.vector_memories):
                                if i not in self.foraged_food_index:
                                    tmp = np.sum(np.abs(f_m - memory))
                                    if tmp < diff:
                                        diff = copy.copy(tmp)
                                        min_ind = copy.copy(i)
                            memory += self.vector_memories[
                                self.current_forage_food_index] - \
                                self.vector_memories[min_ind]
                            memory = np.clip(memory, 0, 1)
                            self.cx.cpu4_mem_gain = 0.005
                            self.current_forage_food_index = copy.copy(min_ind)
                            if log:
                                print('***Start foraging food site {}'.format(
                                    self.food_found_index[min_ind]
                                    + 1))
                    elif dis <= food_radius[ind]:
                        self.current_state = 3
                        self.near_goal_pos = food_sites[ind].copy()
                else:
                    # select the closed food site as the first target
                    diff = []
                    for f_m in self.vector_memories:
                        diff.append(np.sum(np.abs(f_m - memory)))
                    ind = np.argmin(np.array(diff))
                    self.current_forage_food_index = copy.copy(ind)
                    # inverse vector memory
                    memory = (1.0 - self.vector_memories[ind])
                    self.foraging_initialized = True
                    self.cx.cpu4_mem_gain = 0.005
                    if log:
                        print('***Memory based foraging start with food {}'.format(
                            self.food_found_index[ind] + 1))
            elif self.current_state == 3:
                # near food
                if np.sqrt((pos[t+1][0] - self.near_goal_pos[0])**2 +
                           (pos[t+1][1] - self.near_goal_pos[1])**2) <= 1:
                    if self.has_explored:
                        self.current_state = 2
                    else:
                        # got food
                        self.current_state = 1
                        self.step_found_food.append(t)
                        self.vector_memories.append(memory.copy())
                        self.pi_memory_snapshot = memory.copy()
                        remain = list(set(range(len(food_sites)))
                                      - set(self.food_found_index))
                        if log:
                            print('***t={}:got food {}, remain: {} .'.format(
                            t, self.food_found_index[-1]+1, np.array(remain)+1))
            elif self.current_state == 4:
                # near nest
                if np.sqrt((pos[t+1][0] - pos[0][0])**2 +
                           (pos[t+1][1] - pos[0][1])**2) <= 1:
                    if self.has_explored:
                        if log:
                            print('***t={}:task completed'.format(t))
                        self.forage_completed = True
                        break
                    else:
                        # record the timestamp of getting nest
                        self.step_got_home.append(t)
                        # recaliborate the vector memory
                        self.vector_memories[-1] - memory + 0.5
                        if log:
                            print('***t={}:got nest'.format(t))
                        # got all food, exit
                        if len(self.food_found_index) == len(food_sites):
                            self.has_explored = True
                            break
                        # got original site (home)
                        self.current_state = 0
                        memory = np.ones(16)*0.5
                        tb1 = np.zeros(8)
                        self.cx.cpu4_mem_gain = 0.005
            else:
                pass
            
            states.append(self.current_state)
            
        return pos, h, velocity, pi_memory, lgmd_mp, ffi_mp, states
