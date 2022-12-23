import copy
import numpy as np
from scipy.special import expit
import cv2

# Constants
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN1 = 2
N_TN2 = 2
N_CPU4 = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B

# TUNED PARAMETERS:
tl2_slope_tuned = 6.8
tl2_bias_tuned = 3.0

cl1_slope_tuned = 3.0
cl1_bias_tuned = -0.5

tb1_slope_tuned = 5.0
tb1_bias_tuned = 0.0

cpu4_slope_tuned = 5.0
cpu4_bias_tuned = 2.5

cpu1_slope_tuned = 5.0
cpu1_bias_tuned = 2.5

motor_slope_tuned = 1.0
motor_bias_tuned = 3.0


class CX(object):
    """Abstract base class for any central complex model."""

    def __init__(self, tn_prefs=np.pi/4.0,
                 cpu4_mem_gain=0.005):
        self.tn_prefs = tn_prefs
        self.cpu4_mem_gain = cpu4_mem_gain
        self.smoothed_flow = 0

    def tl2_output(self, theta):
        raise NotImplementedError("Subclasses should implement this!")

    def cl1_output(self, tl2):
        raise NotImplementedError("Subclasses should implement this!")

    def tb1_output(self, cl1, tb1):
        raise NotImplementedError("Subclasses should implement this!")

    def get_flow(self, heading, velocity, filter_steps=0):
        """Calculate optic flow depending on preference angles. [L, R]"""
        A = np.array([[np.sin(heading + self.tn_prefs),
                       np.cos(heading + self.tn_prefs)],
                      [np.sin(heading - self.tn_prefs),
                       np.cos(heading - self.tn_prefs)]])
        flow = np.dot(A, velocity)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow

    def tn1_output(self, flow):
        raise NotImplementedError("Subclasses should implement this!")

    def tn2_output(self, flow):
        raise NotImplementedError("Subclasses should implement this!")

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        raise NotImplementedError("Subclasses should implement this!")

    def cpu4_output(self, cpu4_mem):
        raise NotImplementedError("Subclasses should implement this!")

    def cpu1_output(self, tb1, cpu4):
        raise NotImplementedError("Subclasses should implement this!")

    def motor_output(self, cpu1):
        """Positive output means turn left, negative means turn right."""
        raise NotImplementedError("Subclasses should implement this!")


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)


def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([N_TB1, N_TB1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, N_TB1, endpoint=False)) - 1)/2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W


def noisify_weights(W, noise=0.01):
    """Takes a weight matrix and adds some noise on to non-zero values."""
    N = np.random.normal(scale=noise, size=W.shape)
    # Only noisify the connections (positive values in W). Not the zeros.
    N_nonzero = N * W
    return W + N_nonzero


class CXRate(CX):
    """Class to keep a set of parameters for a model together.
    No state is held in the class currently."""
    def __init__(self,
                 noise=0.1,
                 tl2_slope=tl2_slope_tuned,
                 tl2_bias=tl2_bias_tuned,
                 tl2_prefs=np.tile(np.linspace(0, 2*np.pi, N_TB1,
                                               endpoint=False), 2),
                 cl1_slope=cl1_slope_tuned,
                 cl1_bias=cl1_bias_tuned,
                 tb1_slope=tb1_slope_tuned,
                 tb1_bias=tb1_bias_tuned,
                 cpu4_slope=cpu4_slope_tuned,
                 cpu4_bias=cpu4_bias_tuned,
                 cpu1_slope=cpu1_slope_tuned,
                 cpu1_bias=cpu1_bias_tuned,
                 motor_slope=motor_slope_tuned,
                 motor_bias=motor_bias_tuned,
                 weight_noise=0.0,
                 **kwargs):

        super(CXRate, self).__init__(**kwargs)
        # Default noise used by the model for all layers
        self.noise = noise

        # Weight matrices based on anatomy. These are not changeable!)
        self.W_CL1_TB1 = np.tile(np.eye(N_TB1), 2)
        self.W_TB1_TB1 = gen_tb_tb_weights()
        self.W_TB1_CPU1a = np.tile(np.eye(N_TB1), (2, 1))[1:N_CPU1A+1, :]
        self.W_TB1_CPU1b = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 0, 0]])
        self.W_TB1_CPU4 = np.tile(np.eye(N_TB1), (2, 1))
        self.W_TN_CPU4 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            ]).T
        self.W_CPU4_CPU1a = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ])
        self.W_CPU4_CPU1b = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #9
            ])
        self.W_CPU1a_motor = np.array([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
        self.W_CPU1b_motor = np.array([[0, 1],
                                       [1, 0]])

        if weight_noise > 0.0:
            self.W_CL1_TB1 = noisify_weights(self.W_CL1_TB1, weight_noise)
            self.W_TB1_TB1 = noisify_weights(self.W_TB1_TB1, weight_noise)
            self.W_TB1_CPU1a = noisify_weights(self.W_TB1_CPU1a, weight_noise)
            self.W_TB1_CPU1b = noisify_weights(self.W_TB1_CPU1b, weight_noise)
            self.W_TB1_CPU4 = noisify_weights(self.W_TB1_CPU4, weight_noise)
            self.W_CPU4_CPU1a = noisify_weights(self.W_CPU4_CPU1a,
                                                weight_noise)
            self.W_CPU4_CPU1b = noisify_weights(self.W_CPU4_CPU1b,
                                                weight_noise)
            self.W_CPU1a_motor = noisify_weights(self.W_CPU1a_motor,
                                                 weight_noise)
            self.W_CPU1b_motor = noisify_weights(self.W_CPU1b_motor,
                                                 weight_noise)
        # The cell properties (for sigmoid function)
        self.tl2_slope = tl2_slope
        self.tl2_bias = tl2_bias
        self.tl2_prefs = tl2_prefs
        self.cl1_bias = cl1_bias
        self.cl1_slope = cl1_slope
        self.tb1_slope = tb1_slope
        self.tb1_bias = tb1_bias
        self.cpu4_slope = cpu4_slope
        self.cpu4_bias = cpu4_bias
        self.cpu1_slope = cpu1_slope
        self.cpu1_bias = cpu1_bias
        self.motor_slope = motor_slope
        self.motor_bias = motor_bias

    def tl2_output(self, theta):
        """Just a dot product with preferred angle and current heading"""
        output = np.cos(theta - self.tl2_prefs)
        return noisy_sigmoid(output, self.tl2_slope, self.tl2_bias, self.noise)

    def cl1_output(self, tl2):
        """Takes input from the TL2 neurons and gives output."""
        return noisy_sigmoid(-tl2, self.cl1_slope, self.cl1_bias, self.noise)

    def tb1_output(self, cl1, tb1):
        """Ring attractor state on the protocerebral bridge."""
        prop_cl1 = 0.667   # Proportion of input from CL1 vs TB1
        prop_tb1 = 1.0 - prop_cl1
        output = (prop_cl1 * np.dot(self.W_CL1_TB1, cl1) -
                  prop_tb1 * np.dot(self.W_TB1_TB1, tb1))
        return noisy_sigmoid(output, self.tb1_slope, self.tb1_bias, self.noise)

    def tn1_output(self, flow):
        output = (1.0 - flow) / 2.0
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def tn2_output(self, flow):
        output = flow
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def cpu4_update(self, cpu4_mem, tb1, tn1, tn2):
        """Memory neurons update.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        cpu4_mem += (np.clip(np.dot(self.W_TN_CPU4, 0.5-tn1), 0, 1) *
                     self.cpu4_mem_gain * np.dot(self.W_TB1_CPU4, 1.0-tb1))

        cpu4_mem -= self.cpu4_mem_gain * 0.25 * np.dot(self.W_TN_CPU4, tn2)
        return np.clip(cpu4_mem, 0.0, 1.0)

    def cpu4_output(self, cpu4_mem):
        """The output from memory neuron, based on current calcium levels."""
        return noisy_sigmoid(cpu4_mem, self.cpu4_slope,
                             self.cpu4_bias, self.noise)

    def cpu1a_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = np.dot(self.W_CPU4_CPU1a, cpu4) * np.dot(self.W_TB1_CPU1a,
                                                          1.0-tb1)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1b_output(self, tb1, cpu4):
        """The memory and direction used together to get population code for
        heading."""
        inputs = np.dot(self.W_CPU4_CPU1b, cpu4) * np.dot(self.W_TB1_CPU1b,
                                                          1.0-tb1)

        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1_output(self, tb1, cpu4):
        cpu1a = self.cpu1a_output(tb1, cpu4)
        cpu1b = self.cpu1b_output(tb1, cpu4)
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])

    def motor_output(self, cpu1):
        """outputs a scalar where sign determines left or right turn."""
        cpu1a = cpu1[1:-1]
        cpu1b = np.array([cpu1[-1], cpu1[0]])
        motor = np.dot(self.W_CPU1a_motor, cpu1a)
        motor += np.dot(self.W_CPU1b_motor, cpu1b)
        output = (motor[0] - motor[1]) * 0.25  # To kill the noise a bit!
        return output

    def __str__(self):
        return "rate_pholo"


class LobulaGiantMotionDetectorModel:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.size = self.img_width*self.img_height
        # Luminance persistence matrix
        self.luminance_per = 1/(1+np.exp(np.arange(1, 10)))

        self.p = np.zeros([2, self.img_height, self.img_width])
        self.e = np.zeros([self.img_height, self.img_width])
        self.s = np.zeros([self.img_height, self.img_width])
        self.g = np.zeros([self.img_height, self.img_width])

        # lateral inhibition matrix
        self.Wi = np.array([[0.125, 0.25, 0.125],
                           [0.25,   0.,    0.25],
                           [0.125,  0.25, 0.125]])
        self.i = np.zeros([self.img_height, self.img_width])
        self.i_w = 0.4

        # G layer convolution mask
        self.We = np.ones([3, 3])/9.0
        self.Cde = 0.5
        self.thr_de = 15
        self.lgmd = 0

        # FFI
        self.ffi_thr = 0.5
        self.ffi_spikes = []
        self.ffi_thr_initial = 1.0
        self.alpha_ffi = 0.02

        # LGMD threshold and spiking
        self.lgmd_spikes = []
        self.lgmd_thr = 0.9
        self.spike_interval = 5
        # adaptable threshold
        self.pie_upper = 230
        self.pie_lower = 180
        self.lto_thr = 0
        self.delta_thr_lt = 0.03
        self.alpha_l = 1
        self.alpha_lt = 4
        self.alpha_mp = 1
        self.mp_thr = 0.86

        # model output
        self.spikes = []

        self.frame = np.zeros([2, self.img_height, self.img_width])

    def reset(self):
        self.frame = np.zeros([2, self.img_height, self.img_width])
        self.p = np.zeros([2, self.img_height, self.img_width])
        self.e = np.zeros([self.img_height, self.img_width])
        self.i = np.zeros([self.img_height, self.img_width])
        self.s = np.zeros([self.img_height, self.img_width])
        self.g = np.zeros([self.img_height, self.img_width])
        self.lgmd = 0
        self.lgmd_spikes = []
        self.ffi_spikes = []
        self.spikes = []

    def run(self, luminance_per_num=1,
            fixed_ts=False,
            fixed_ffi_thr=False):
        self.p[1] = self.frame[1] - self.frame[0]
        self.e = copy.deepcopy(self.p[1])
        self.i = cv2.filter2D(self.p[0], -1, self.Wi)
        self.s = self.e - self.i * self.i_w
        Ce = cv2.filter2D(self.s, -1, self.We)
        w = 0.01 + np.max(np.abs(Ce / 4.0))
        self.g = self.s * Ce / w
        self.g[self.g*self.Cde < self.thr_de] = 0
        self.lgmd = 1/(1 + np.exp(-np.sum(self.g)/self.size))
        # FFI - one frame delayed
        self.ffi = np.sum(self.p[0])/(self.size)
        if not fixed_ffi_thr:
            self.ffi_thr = self.ffi_thr_initial + self.alpha_ffi * self.ffi_thr
        self.ffi_spikes.append(1 if self.ffi >= self.ffi_thr else 0)

        self.p[0] = self.p[1]

        # spiking
        if not fixed_ts:
            # adaptable Ts using FFM cell
            Ld = (np.sum(np.max(self.frame[1], axis=0)) +
                  np.sum(np.max(self.frame[1], axis=1)))\
                / (self.img_height + self.img_width)
            if Ld > self.pie_upper:
                thr_lt = self.delta_thr_lt + self.alpha_l * self.delta_thr_lt
            elif Ld < self.pie_lower:
                thr_lt = self.delta_thr_lt - self.alpha_l * self.delta_thr_lt
            else:
                thr_lt = 0
            self.lgmd_thr = self.alpha_lt * thr_lt + \
                self.alpha_mp * self.mp_thr

        self.lgmd_spikes.append(1 if self.lgmd >= self.lgmd_thr else 0)
        
        # final output
        tmp = sum(self.lgmd_spikes[-self.spike_interval:])
        self.spikes.append(1 if (self.ffi_spikes[-1] == 0)
                           and (self.lgmd_spikes[-1] == 1) 
                           and (tmp == self.spike_interval)
                           else 0)

        return self.e, self.i, self.s, Ce, self.g, self.lgmd, self.ffi
    
    def run_no_enhance(self, luminance_per_num=1,
                       fixed_ts=False,
                       fixed_ffi_thr=False):
        self.p[1] = self.frame[1] - self.frame[0]

        self.i = cv2.filter2D(self.p[0], -1, self.Wi)
        self.e = self.p[1] - self.i * self.i_w
        Ce = cv2.filter2D(self.e, -1, self.We)
        w = 0.01 + np.max(np.abs(Ce / 4.0))
        self.s = np.abs(self.e) * Ce / w
        self.s[self.s < 0] = 0
        self.lgmd = 1/(1 + np.exp(-np.sum(self.s)/self.size))
        # FFI - one frame delayed
        self.ffi = np.sum(self.p[0])/(self.size)
        if not fixed_ffi_thr:
            self.ffi_thr = self.ffi_thr_initial + self.alpha_ffi * self.ffi_thr
        self.ffi_spikes.append(1 if self.ffi >= self.ffi_thr else 0)

        self.p[0] = self.p[1]

        # spiking
        if not fixed_ts:
            # adaptable Ts using FFM cell
            Ld = (np.sum(np.max(self.frame[1], axis=0)) +
                  np.sum(np.max(self.frame[1], axis=1)))\
                / (self.img_height + self.img_width)
            if Ld > self.pie_upper:
                thr_lt = self.delta_thr_lt + self.alpha_l * self.delta_thr_lt
            elif Ld < self.pie_lower:
                thr_lt = self.delta_thr_lt - self.alpha_l * self.delta_thr_lt
            else:
                thr_lt = 0
            self.lgmd_thr = self.alpha_lt * thr_lt + \
                self.alpha_mp * self.mp_thr

        self.lgmd_spikes.append(1 if self.lgmd >= self.lgmd_thr else 0)
        
        # final output
        tmp = sum(self.lgmd_spikes[-self.spike_interval:])
        self.spikes.append(1 if (self.ffi_spikes[-1] == 0) \
                           and (tmp == self.spike_interval)
                           else 0)

        return self.e, self.i, self.s, Ce, self.s, self.lgmd, self.ffi
