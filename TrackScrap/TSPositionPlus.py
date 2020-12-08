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


    def get_burstCoastCharacter(self, friction=None):
        # modifying acc_s such that burst is if: ds/dt + |dphi/dt| * s > 0
        # instead of Burst if: ds/dt > 0
        burstPlus_coastMinus = np.sign( np.sign(self.acc_s) - 0.5)
        # update properties:
        acc_true = np.sqrt(self.acc_v**2 +
                           self.acc_phi**2) * burstPlus_coastMinus
        (self.up_rate, self.up_len,
         self.up_acc) = get_update_rate(acc_true)
        if friction is not None:
            acc_corrected = np.sqrt((self.acc_s + self.s * friction)**2 +
                                    (self.d_phi * self.s)**2) * burstPlus_coastMinus
            (self.up_rateSmoo, self.up_len,
             self.up_accCorrected) = get_update_rate(acc_corrected)
        # coast behavior:
        self.coast_acc, self.coast_s = get_coast_behavior(burstPlus_coastMinus,
                                                          self.acc_s, self.s)
        # burst behavior:
        (self.burstStart, self.burstEnd,
         self.bursterID) = get_burst_blockDetails(burstPlus_coastMinus)
        (self.burst_acc_s, self.burst_acc_v, self.burst_acc_phi,
         self.burst_s, self.burst_d_phi
         ) = get_burst_behavior(burstPlus_coastMinus,
                                self.acc_s, self.acc_v,
                                self.acc_phi, self.s, self.d_phi)


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


def get_coast_behavior(burstPlus_coastMinus, acc_s, s):
    '''
    INPUT:
        acc_s shape=(time, N)
            acceleration
        s shape=(time, N)
            speed
    OUTPUT:
        coast_acc list
            accelerations at coasting
        coast_s list
            speed at coasting
    '''
    assert len(acc_s.shape) == 2, 'len(acc_s.shape) != 2'
    time, N = acc_s.shape
    coast_acc = []
    coast_s = []
    for i in range(N):
        thereLow = gen.find_blocks_low(burstPlus_coastMinus[:, i], 0, 1, 1, noBlocks=True)
        if len(thereLow) > 0:
            coast_acc.append(list(acc_s[thereLow, i]))
            coast_s.append(list(s[thereLow, i]))
    return coast_acc, coast_s


def get_burst_blockDetails(burstPlus_coastMinus):
    '''
    in contrast to get_burst_behavior it just returns
    the start-, end-time and the ID of the bursting agent
        This allows in combination with the other measures
        to estimate the social forces
    INPUT:
        burstPlus_coastMinus shape=(time, N)
            timeseries which marks a burst with +1 and coast with -1
    OUTPUT:
        burstBlockDetails list [[t_start, t_end, IDburst], ...]
            list of block-detail-lists where the latter defines start
            and end time and ID of burster
    '''
    assert len(burstPlus_coastMinus.shape) == 2, 'Invalid Shape: len(shape) != 2'
    time, N = burstPlus_coastMinus.shape
    burstStart = []
    burstEnd = []
    bursterID = []
    for i in range(N):
        blocksLarge = gen.find_blocks_large(burstPlus_coastMinus[:, i], 0, 1, 1, noBlocks=False)
        if len(blocksLarge) > 0:
            blocksLarge = np.array(blocksLarge).T
            burstStart += list(blocksLarge[0])
            burstEnd += list(blocksLarge[1])
            bursterID += len(blocksLarge[0]) * [i]
    return burstStart, burstEnd, bursterID


def get_burst_behavior(burstPlus_coastMinus, acc_s, acc_v, acc_phi, s, d_phi):
    '''
    INPUT:
        burstPlus_coastMinus shape=(time, N)
            timeseries which marks a burst with +1 and coast with -1
        acc_s shape=(time, N)
            acceleration
        s shape=(time, N)
            speed
    OUTPUT:
        coast_acc list
            accelerations at coasting
        coast_s list
            speed at coasting
    '''
    assert len(acc_s.shape) == 2, 'len(acc_s.shape) != 2'
    time, N = acc_s.shape
    burst_acc_s = []
    burst_acc_v = []
    burst_acc_phi = []
    burst_s = []
    burst_d_phi = []
    for i in range(N):
        thereLarge = gen.find_blocks_large(burstPlus_coastMinus[:, i], 0, 1, 1, noBlocks=True)
        if len(thereLarge) > 0:
            burst_acc_s.append(list(acc_s[thereLarge, i]))
            burst_acc_v.append(list(acc_v[thereLarge, i]))
            burst_acc_phi.append(list(acc_phi[thereLarge, i]))
            burst_s.append(list(s[thereLarge, i]))
            burst_d_phi.append(list(d_phi[thereLarge, i]))
    return burst_acc_s, burst_acc_v, burst_acc_phi, burst_s, burst_d_phi


def get_update_rate(acc_s):
    '''
    returns the update events, length and the 
    "update" (marked by 1st. positive acceleration)
    estimated in window of size "win"
    INPUT:
        acc_s.shape(time, N)
            change of speed NOT of velocity-vector
            OR
            CLASS with attr. acc_s
        win float
    OUTPUT:
        update.shape(time, N)
    '''
    if hasattr(acc_s, 'acc_s'):
        acc_s = acc_s.acc_s
    assert len(acc_s.shape) == 2, 'len(acc_s.shape) != 2'
    time, N = acc_s.shape

    update = np.zeros((time, N), dtype='float')
    upLen = []
    upAcc = []
    for i in range(N):
        blocks = gen.find_blocks_large(acc_s[:, i], 0, 1, 1)
        if len(blocks) > 0:
            update[blocks[:, 0], i] = 1     # = update time series
            temp = list(blocks[:, 1] + 1 - blocks[:, 0])
            upLen.append(temp)
            temp = [acc_s[b[0]:b[1]+1].mean() for b in blocks]
            upAcc.append(temp)
    return update, upLen, upAcc
