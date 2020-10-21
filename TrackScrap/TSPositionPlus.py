import numpy as np
import matplotlib.pyplot as plt
from TrackScrap import general as gen
from TrackScrap import TSPosition as tsp
from sklearn.neighbors import KDTree


class TSplus:
    '''
    creates from position time-series different individual 
    and collective time-series
    INPUT:
        pos.shape(time, N, 2)
        window int
            needed for computation of update_rate(window of average)
            50 suggested 
        smoothed boolean
    '''
    def __init__(self, pos, width, smoothed):
        if len(pos.shape) == 2:
            pos = pos.reshape(-1, 1, 2)
        Nnans = np.isnan(pos).sum()
        if Nnans > 0:
            print('ATTENTION raw data has nans: ', Nnans)
        self.time, self.N, _ = pos.shape
        if smoothed:
            smodata = pos.reshape(pos.shape[0], -1)
            smodata = gen.smooth2D(smodata, width)
            pos = smodata.reshape(pos.shape)
        dat = tsp.pos2vel_acc(pos)
        self.pos = dat[:, :, :2]
        self.vel = dat[:, :, 2:4]
        self.acc = dat[:, :, 4:6]
        self.phi = dat[:, :, 6]
        self.acc_v = dat[:, :, 7]
        self.acc_phi = dat[:, :, 8]
        self.acc_s = dat[:, :, 9]
        self.d_phi = dat[:, :, 10]
        self.s = np.sqrt(self.vel[:, :, 0]**2 + self.vel[:, :, 1]**2)
        if self.N > 1:
            self.get_collective_TS()


    def get_collective_TS(self):
        self.IID = tsp.get_IID(self.pos)
        self.nnd = tsp.get_NND(self.pos)
        self.Area_ConvexHull = tsp.get_ConvexHull_Area(self.pos)
        self.OP_pol = get_polarization(self.phi)
        self.Group_speed = get_group_speed(self.vel)


def find_global_AccArtefact(acc_s, sigma):
    '''
    finds timepoints where many particles suddenly 
    strongly accelerate and in the NEXT timestep suddenly decelerate
    If this acceleration artefact happens simultaneously for ALL particles
        -> the camera has probably dropped frames
    INPUT:
        acc_s.shape(time, N)
            change of speed NOT of velocity-vector
            OR
            CLASS with attr. acc_s
        sigma float
            how many standard deviation are considered
    '''
    if hasattr(acc_s, 'acc_s'):
        acc_s = acc_s.acc_s
    assert len(acc_s.shape) == 2, 'len(acc_s.shape) != 2'
    assert sigma > 0, 'sigma <= 0'
    deltaAcc = np.diff(acc_s[:, :, 7], axis=0)
    deltaAcc = deltaAcc.sum(axis=1) # interested in simultaneous events
    avg = deltaAcc[deltaAcc < 0].mean()
    sig = deltaAcc[deltaAcc < 0].std()
    fdrops = np.where(deltaAcc < (avg - sigma * sig))[0]
    return fdrops


def get_polarization(phi):
    '''
    find macroscopic measures of collective (polarization, ...)
    INPUT:
        phi.shape(time, N)
            heading of agent 
        OR
        class with attr phi
    OUTPUT:
        list of timeseries of macroscopic variables
    '''
    if hasattr(phi, 'phi'):
        phi = phi.phi
    assert len(phi.shape) == 2, 'len(dat.shape) != 2'
    time, N = phi.shape

    polOrder = np.empty(time, dtype='float')
    for i in range(time):
        polOrder[i] = np.sqrt(np.cos(phi[i, :]).sum() ** 2 +
                              np.sin(phi[i, :]).sum() ** 2) / N
    return polOrder


def get_group_speed(vs):
    '''
    INPUT:
        vs.shape(Time, N, 2)
    '''
    v = vs.mean(axis=1)
    v = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    return v


def standardizePV(pos, vel):
    '''
    sets the COM in the center and rotates 
    such that AVG_vel goes in y-direction
    INPUT:
        pos.shape (T, N, 2)
        vel.shape (T, N, 2)
    OUTPUT:
        pos.shape (T, N, 2)
        vel.shape (T, N, 2)
    '''
    grp_vel = np.nanmean(vel, axis=1)
    grp_phi = np.arctan2(grp_vel[:, 1], grp_vel[:, 0])
    # COM at (0, 0):
    com = np.nanmean(pos, axis=1)
    pos_out = pos - com[:, None, :]
    # Rotation such that grp_vel = y-direction
    vel_out = vel * 1
    phi_ccw = np.pi/2 - grp_phi   
    for i, phi in enumerate(phi_ccw):
        pos_out[i] = gen.RotateCcw(pos_out[i], phi)
        vel_out[i] = gen.RotateCcw(vel[i], phi)
    return pos_out, vel_out
