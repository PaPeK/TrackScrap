import numpy as np
import os
from TrackScrap import general as gen
from TrackScrap import TSPosition as tsp
from TrackScrap import TSPositionPlus as tspp

class IDTracker_TS:
    '''
    name_file
    time
    N
    dat     shape(time, N, 2)
    prob    shape(time, N)
    valid data blocks:
        v_block     shape(v_N, 2)
            v_N valid blocks, 1 block contains start- and end-time
            of valid data
        v_N         int
            # of valid blocks
        v_time     shape(v_N)
            time-length of each block
    '''
    def __init__(self, name_file, ID=None):
        self.name_file = name_file
        self.load_dat()
        if ID is not None:
            self.ID = ID
            self.onlyDatOfId()

    def load_dat(self):
        ext = os.path.splitext(self.name_file)[-1]
        if ext == '.txt':
            dat = np.loadtxt(self.name_file, skiprows=1)
        elif ext == '.csv':
            dat = gen.loadCsv(self.name_file, skiprows=1, skipcols=1)
        self.time = dat.shape[0]
        self.N = int(dat.shape[1]/3)
        liste = list(zip(range(0, self.N*3, 3), range(1, self.N*3, 3)))
        liste = np.array(liste).flatten()
        self.prob = dat[:, list(range(2, self.N*3, 3))]
        dat = dat[:, liste]
        self.dat = dat.reshape(-1, self.N, 2)


    def onlyDatOfId(self):
        assert self.ID <= self.N, 'ID <= self.N: {} <= {}'.format(self.ID, self.N)
        self.N = 1
        self.prob = self.prob[:, self.ID].reshape(self.time, 1)
        self.dat = self.dat[:, self.ID].reshape(self.time, 1, 2)


    def exclude_nans(self):
        multiple_undetected = tsp.Multiple_Nans(self.dat)[0]
        if len(multiple_undetected) == 0:
            self.make_v_block()
        else:
            blocks = gen.get_blocks(multiple_undetected, 1)
            self.exclude_blocks(blocks)


    def exclude_jumps(self, speed_max, speed_mean, returnBlocks=None):
        '''
        CAUSED: by missing frames or id-switch
        IDENTIFIED: by an extreme positive acceleration
                    directly followed by extreme negative acceleration
                    -> acc = [80, -80]
        extreme acceleration= speed_max - speed_mean
        '''
        if returnBlocks is None:
            returnBlocks = False
        vel = np.diff(self.dat, axis=0)
        v = np.sqrt(np.sum(vel**2, axis=2))
        acc = np.diff(v, axis=0)
        jump = -1 * np.diff(acc, axis=0)
        crazy_jump = (speed_max - speed_mean)*2
        block = gen.find_blocks_large(jump, crazy_jump, 1, 1)
        if len(block) > 0:
            # block refers to acceleration: positions 2 frames later needs inclusion
            block[:, 1] += 2
            self.exclude_blocks(block)
            if returnBlocks:
                return block


    def exclude_jumps2(self, returnBlocks=None):
        '''
        CAUSED: by missing frames or id-switch
        IDENTIFIED: (i) by an positive acceleration
                        directly followed by a similar negative acceleration
                        -> acc = [80, -80]
                    (ii) the acceleration should have doubled the speed
        '''
        if returnBlocks is None:
            returnBlocks = False
        vel = np.diff(self.dat, axis=0)
        v = np.sqrt(np.sum(vel**2, axis=2))
        acc = np.diff(v, axis=0)
        case1 = -acc[:-1] / acc[1:]
        case1 = (0.8 < case1) & (case1 < 1.2) & (acc[:-1] > 0)
        # check case (ii)
        case2 = (acc[:-1] / v[:-2]) > 2
        jump = case1 & case2
        block = gen.find_blocks_large(jump.astype(int), 0.5, 1, 1)
        if len(block) > 0:
            # block refers to acceleration: positions 2 frames later needs inclusion
            block[:, 1] += 3
            self.exclude_blocks(block)
            if returnBlocks:
                return block


    def exclude_zigZag(self, vlimit, angle=None, returnBlocks=None):
        '''
        CAUSED: by tracking errors
        Reasoning why it must be an error:
            -Fish can have at certain speeds a maximal turning ability.
            -Of course, a startle can exceed this maneuvrability but only for 1 frame
            -For 2 frames followed it is not realistic
        IDENTIFIED: (i) the direction changes stronger as an threshold
                        angle twice
                    (ii) all direction changes happen velocities faster
                        than the limiting speed
        TODO: actually it is not checked if the second direction change goes
                in the opposite direction as the first
        '''
        if returnBlocks is None:
            returnBlocks = False
        if angle is None:
            angle = np.pi/2
        vel = np.diff(self.dat, axis=0)
        v = np.sqrt(np.sum(vel**2, axis=2))
        # check case (i)
        case1 = np.array([gen.angle_between(vel[:-1, i], vel[1:, i])
                          for i in range(self.N)]).T
        case1 = (case1[:-1] > angle) & (case1[1:] > angle)
        # check case (ii)
        case2 = (v[:-2] > vlimit) & (v[1:-1] > vlimit) & (v[2:] > vlimit)
        zigZag = case1 & case2
        block = gen.find_blocks_large(zigZag.astype(int), 0.5, 1, 1)
        if len(block) > 0:
            # block refers to acceleration: positions 2 frames later needs inclusion
            block[:, 1] += 3
            self.exclude_blocks(block)
            if returnBlocks:
                return block



    def interpolate_valid_blocks(self):
        '''
        position interpolation between valid data-blocks
        '''
        assert self.ID <= self.N, 'ID <= self.N: {} <= {}'.format(self.ID, self.N)
        if len(self.v_time) < 2:
            return
        for i in range(len(self.v_block)-1):
            i_start = self.v_block[i][1]
            i_end = self.v_block[i+1][0]
            tsp.Interpolate(self.dat[i_start:i_end+1, 0])


    def exclude_low_prob(self, minp, minNr=None):
        '''
        data where at least "minNr" individual has prob < minp
        is added to excluded blocks (self.ex_block)
        '''
        if minNr is None:
            minNr=1
        block = gen.find_blocks_low(self.prob, minp, minNr, 1)
        if len(block) > 0:
            self.exclude_blocks(block)


    def exclude_too_short(self, minlen):
        '''
        valid blocks with v_time < minlen -> excluded
        '''
        if len(self.v_block) > 0:
            block = self.v_block[self.v_time < minlen]
            if len(block) > 0:
                self.exclude_blocks(block)


    def exclude_blocks(self, block):
        '''
        add block/blocks to self.ex_blocks and directly generates
        blocks of valid data (self.v_blocks)
        '''
        if hasattr(self, 'ex_block'):
            self.ex_block = np.vstack((self.ex_block, block))
            self.ex_block = gen.clean_blocks(self.ex_block)
        else:
            self.ex_block = np.array(block).reshape(-1, 2)
        self.make_v_block()


    def make_v_block(self):
        '''
        computes from excluded blocks the valid blocks:
        e.g.:
            ex_block = [[0, 10], [100, 140], [end-3, end]]
            ->
            v_block = [[11, 99], [141, end-4]]
        '''
        if hasattr(self, 'ex_block'):
            # if whole block is excluded
            if self.ex_block[0, 0] == 0 and self.ex_block[0, 1] == self.time - 1:
                self.v_block = np.zeros((0, 2), dtype='int')
                self.v_time = np.array([0])
                self.v_N = len(self.v_block)
                return
            valid = np.ones(self.time)
            for ex in self.ex_block:
                valid[ex[0]:ex[1]+1] = 0
            self.v_block = gen.find_blocks_large(valid, 0.5, 1, 1)
        else:
            self.v_block = np.array([[0, self.time - 1]])
        self.v_time = (np.diff(self.v_block, axis=1) + 1).flatten()
        self.v_N = len(self.v_block)


    def make_data_plus(self, win, smooth):
        '''
        executes for each block an function which returns and object which is saved in a list
        INPUT:
            win float
                window of smoothening function
            smooth boolean
                if smoothed the 
        '''
        self.exclude_nans()
        self.exclude_too_short(3) # minimum frames for acceleration
        self.exclude_too_short(2*win)
        self.Dat = []
        for b in self.v_block:
            ts_plus = tspp.TSplus(self.dat[b[0]:b[1]+1], win, smooth)
            self.Dat.append(ts_plus)


    def merge_Dat(self, attri, splitByNan=None):
        if splitByNan is None:
            splitByNan = False
        assert hasattr(self.Dat[0], attri), '{} is not an attribute of the class'.format(attri)
        merged = getattr(self.Dat[0], attri)
        if type(merged) == np.ndarray:
            nana = np.empty([1] + list(merged.shape[1:])) * np.nan
            for i in range(1, self.v_N):
                if splitByNan:
                    merged = np.append(merged, nana, axis=0)
                merged = np.append(merged, getattr(self.Dat[i], attri), axis=0)
        else: # assume list used because unequal length in between agents
            for i in range(1, self.v_N):
                merged += getattr(self.Dat[i], attri)
        return merged
