import numpy as np
from numpy import random
from matplotlib import pyplot as plt

from utils.AngleWrap import AngleWrapList
from utils.tcomp import tcomp

class FOVSensor():
    def __init__(self, cov_sensor, fov, max_range):
        self.fov = fov
        self.max_range = max_range
        self.cov_sensor = cov_sensor

    def observe(self, from_pose, world, noisy=True, draw=False):
        delta = world - from_pose[0:2]

        z = np.empty((2, world.shape[1]))
        z[0, :] = np.sqrt(np.sum(delta**2, axis=0))
        z[1, :] = np.arctan2(delta[1, :], delta[0, :]) - from_pose[2, 0]
        z[1, :] = AngleWrapList(z[1, :])

        if noisy: 
            z += np.sqrt(self.cov_sensor)@random.rand(2, world.shape[1])
            
        if draw:
            self.drawLines(from_pose, world)

        return z

    def random_observation(self, from_pose, world, noisy=True, fov=True):
        if fov:
            z, feats_idx = self.observe_in_fov(from_pose, world)
        else:
            z = self.observe(from_pose, world, noisy=noisy)

        n_landmarks = z.shape[1]

        if n_landmarks > 0:
            rand_idx = random.randint(n_landmarks)

            z = z[:, [rand_idx]]
            if fov:
                rand_idx = feats_idx[rand_idx]

            return z, rand_idx
        else:
            return z, -1

    def observe_in_fov(self, from_pose, world, noisy=True):
        ang_lim = self.fov/2

        z = self.observe(from_pose, world, noisy)

        feats_idx = np.where((np.abs(z[1, :] < ang_lim) & (z[0, :] < self.max_range)))[0]
        z = z[:, feats_idx]
        
        return z, feats_idx

    def drawFOV(self, fig, ax, from_pose, color='b', linewidth=.5, **kwargs):
        alpha = self.fov/2
        angles = np.linspace(-alpha,alpha, (self.fov/0.01))
        nAngles = angles.shape[0]
        arc_points = np.zeros((2,nAngles))
        
        for i in range(nAngles):
            u = np.vstack([
                self.max_range*np.cos(angles[i]),
                self.max_range*np.sin(angles[i]),
                1
            ])
            aux_point = tcomp(from_pose,u)
            arc_points[0,i] = aux_point[0,0]
            arc_points[1,i] = aux_point[1,0]
            
        h = ax.plot(
                np.hstack((from_pose[0,0], arc_points[0], from_pose[0,0])),
                np.hstack((from_pose[1,0], arc_points[1], from_pose[1,0])),
                color=color,
                linewidth=linewidth,
                **kwargs
            )

        return h
    
    def drawLines(self, fig, ax, from_pose, world):
        for i in range(world.shape[1]):
            if i >=0:
                ax.plot(
                    [from_pose[0, 0], world[0, i]],
                    [from_pose[1, 0], world[1, i]],
                    linestyle=':'
            )