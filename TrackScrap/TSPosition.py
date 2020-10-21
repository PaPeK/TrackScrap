import numpy as np
import matplotlib.pyplot as plt
from TrackScrap import general as gen
from scipy.spatial import ConvexHull


# def Multiple_Undetected_F(dat):
def Multiple_Nans(dat, verbose=False):
    '''
    returns an an array which contains the times at which
    at least one Fish(F) was not detected by IDtracker 
    and times at which 
    different F were not detected simultaneously
    INPUT:
        dat.shape(time, N, 2) or (time, 2)
    OUTPUT:
        result list, len(result)=N
            each ellement contains array of times where:
            result[0] at least 1 F has x=nan y=nan
            result[1] 1 F has x=nan y=nan
            result[2] 2 F has x=nan y=nan
            ...
            result[N] N F has x=nan y=nan
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    time = dat.shape[0]
    if len(dat.shape) < 3:
        dat = 1*dat 
        dat = dat.reshape(time, 1, 2)
    N = dat.shape[1]
    # multiple_nans = (np.isnan(dat.reshape(time,N*2)).sum(axis=1))).astype('int')
    multiple_nans = np.isnan(dat[:, :, 0]).sum(axis=1)
    result = np.zeros(N+1, dtype='object')
    result[0] = np.where(multiple_nans!=0)[0]
    if verbose:
        print('multiple_nans.shape: ', multiple_nans.shape)
        print("at least 1 F is not detected:",
              len(result[0]), " frames")
    for i in range(1, N+1):
        result[i] = np.where(multiple_nans==i)[0]
        if verbose:
            print(i, " fish are simultaneous not detected in ",
                  len(result[i]), " frames")
    return result

# def SortLastPosByError(dat):
def switch_id_by_error(dat):
    '''
    sorting the last position of agents by comparing it with 
    the predicted positions of all agents 
    INPUT: 
        dat.shape(time, N, 2) last 2 are x, y
    OUTPUT:
        result.shape(N, 2)
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 3, 'len(dat.shape) != 3'
    predictors = dat[:-1]
    values = dat[-1]
    N = len(values)
    errors = np.zeros((N, N))
    acceptableErr = np.zeros(N)
    result = np.zeros((N, 2))
    
    # not able to use
    pastVel = np.diff(predictors, axis=0)
    avgVel = np.mean(pastVel, axis=0)
    predictions = predictors[-1, :] + avgVel 
    
    # creating error matrix
    for i in range(N):   # predictions of agent i
        acceptableErr[i] = 0.5 * np.sqrt(np.dot(avgVel[i], avgVel[i]))
        for j in range(N):   # compared with values of agent j
            diff = predictions[i] - values[j]
            errors[i, j] = np.sqrt(np.dot(diff, diff))
    
    switched = False
    maxxe = errors.max() + 1
    # rewriting errors: if error is acceptable -> set to max-error all errors concerning
    for i in range(N):
        if errors[i, i] <= acceptableErr[i]:
            errors[i, :] = maxxe
            errors[:, i] = maxxe
            errors[i, i] = 0
            errors[i, i] = 0
    # sorting with smallest error:
    for i in range(N):
        minne = errors.min() 
        there = np.where(errors == minne)
        there0 = 1*there[0]
        there1 = 1*there[1]
        # if error same for multiple combinations -> try to keep index
        if len(there[0]) > 1:
            for k in range(len(there[0])):
                if there[0][k] == there[1][k]:
                    there0 = there[0][k]
                    there1 = there[1][k]
                    # print('multiple choices possible: keep ID possible')
                    break
                if k == len(there[0])-1:
                    there0 = there[0][0]
                    there1 = there[1][0]
                    print('multiple choices possible: random choice')
            
        result[there0] = values[there1]
        errors[there0, :] = maxxe
        errors[:, there1] = maxxe
        try:
            if there0 != there1:
                switched = True
        except:
            print('there ', there)
        
    return result, switched

# def SwitchIdInBlock(dat, prob, block, win, minP):
def switch_id_inblock(dat, prob, block, win, minP):
    '''
    switch IDs in whole time-block, identified by low id-probability
    of more than 1 agent
    IDs only switched between agents with id-probability ("prob") < minP
    IDs switching based on predictions of last "win" time-step before
    INPUT:
        dat.shape(time, N, 2)
            whole dat, with last 2 as x, y
        prob.shape(time, N)
            probabilities of correct ID
        block.shape(2) OR {M, 2}
            start- and end-time of block where at LEAST 2 agents have prob<minP
        win 
            window used for predictions
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 3, 'len(dat.shape) != 3'
    time, N, var = dat.shape
    if len(block.shape) < 2:
        assert len(block) == 2, 'Wrong dimension of blocks'
        block = block.reshape(1,2)
    M = len(block)
    switch = np.zeros(time, dtype='bool')
        
    for i in range(M):
        ts = block[i, 0]
        assert ts > 0, 'attention first block starts at t=0 -> no past to predict'
        te = block[i, 1]
        for t in range(ts, te+1):
            lowP = np.where(prob[t] < minP)[0]
            assert len(lowP) >= 2, 'Blocks must contain at least 2 agents with prob<minP'
            
            # if past is shorter than required window -> reduce window:
            window = 1*win
            if t-win < 0:
                window += (t - win)
                
            # sort by smallest error of prediction:
            dat[t, lowP], switch[t] = switch_id_by_error(dat[t-window:t+1, lowP]) 
    return switch
            
# def randomizeIds(dat):
def randomize_id(dat):
    '''
    randomizes the Ids of the trajectories
    INPUT:
        dat.shape(time, N, 2)
            N=# of particles and last 2 as x, y
    OUTPUT:
        prob.shape(time, N)
            probability of particle having correct ID
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 3, 'len(dat.shape) != 3'
    time, N, v = dat.shape
    prob = np.ones((time, N), dtype='float')
    for i in range(time):
        r1 = int(np.random.uniform(2, N))
        r2 = np.random.choice(N, r1, replace=False)
        s = np.sort(r2)
        dat[i, s] = 1* dat[i, r2]
        prob[i, s] = 0
    return prob

# def TestSwitchIdInBlock(N, steps, M):
def TEST_switch_id_inblock(N, steps, M):
    '''
    test if fct. SwitchIdInBlock works properly by 
    generating linear positions, shuffling them at specific parts and marking
    it with a ID-probability prob=0.
    Then the trajectories are corrected and corrections compared to real dat.
    Note:
        works only for strongly linear cases 
            -> strong curvature in trajectory poses problems
        Improve: not only check if trajectory is consistent with past but also 
                    with future tracks!
    INPUT:
        N int
            number of particles
        steps int
            number of timesteps
        M int
            number of blocks in which IDs are switched
    '''
    # linear trajectories computed:
    xspeed = np.random.choice(np.arange(0.5, 1.5, step=0.1), N)
    # xspeed = np.ones(N)
    slope = np.random.choice(np.arange(-10, 10), N, replace=False)
    ystart = np.random.choice(np.arange(-10*steps, 10*steps), N)
    print('ystart: ', ystart)
    dat = np.zeros((steps, N, 2))
    f, axs = plt.subplots()
    sigma = 0.001
    for i in range(N):
        x = np.linspace(0, steps * xspeed[i], steps) + sigma * np.random.standard_normal(steps)
        y = slope[i] * x + ystart[i] + sigma * np.random.standard_normal(steps)
        dat[:, i, 0] = x
        dat[:, i, 1] = y
        axs.plot(x, y)
    f.show()
    
    # create random blocks
    blocks0 = np.random.choice(steps, M*2, replace=False)
    blocks0 = np.sort(blocks0).reshape(M, 2)
    print('blocks0', blocks0)
    
    # randomize the dat in the blocks
    prob = np.ones((steps, N), dtype='float')
    for i in range(M):
        prob[blocks0[i, 0]:blocks0[i, 1]+1] = randomize_id(dat[blocks0[i, 0]:blocks0[i, 1]+1])
        
    # show randomized dat
    f, axs = plt.subplots()
    for i in range(N):
        axs.plot(dat[:, i, 0], dat[:, i, 1])
    f.show()
    
    # TEST STARTS:
    # find blocks
    minP = 0.5
    minNr = 2
    minBlock = 1
    blocks = gen.find_blocks_low(prob, minP, minNr, minBlock)
    print('blocks', blocks)
    print('Block finding Error:', (blocks -blocks0).sum())
    
    
    # Correct trajectories
    win = 10
    switch = SwitchIdInBlock(dat, prob, blocks, win, minP)
    print('at ', switch.sum(), 'different times happend at least 2 switches')
    
    # show randomized dat
    f, axs = plt.subplots()
    for i in range(N):
        axs.plot(dat[:, i, 0], dat[:, i, 1])
    f.show()


# def Interpolate(dat):
def Interpolate(dat):
    '''
    substitutes points in between 1. and last by linear interpolations
    INPUT:
        dat.shape(time, 2)
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 2, 'len(dat.shape) != 2'
    steps = len(dat) - 2
    x0 = dat[0]
    x1 = dat[-1]
    v01 = x1 - x0
    r01 = np.sqrt(np.dot(v01, v01))
    v01 /= r01
    dist = r01/(float(steps) + 1)
    for i in range(1, steps+1):
        dat[i] = x0 + i * dist *v01


def pos2vel_acc(dat):
    '''
    computes from position data segments the velocity, accelaration
    and oientation 
    INPUT:
        dat: (time, particle, x-y)
    OUTPUT:
        out: (time, particle, x-y-vx-vy-ax-ay-phi-a_v-a_phi-a_s-d_phi)
    '''

    time, N, _ = dat.shape
    output = np.zeros((time-2, N, 11), dtype='float')

    for i in range(N):
        pos = dat[:, i, :]
        vel = np.diff(pos, axis=0)
        phi = (np.arctan2(vel[:, 1], vel[:, 0])
                           + 2 * np.pi) % (2 * np.pi)
        acc = np.diff(vel, axis=0)
        s = np.sqrt(np.sum(vel**2, axis=1))
        a_s = np.diff(s)
        u = vel / s[:, None] 
        # correction for s=0
        u[s == 0] = 0
        u_perp = np.empty_like(u)
        u_perp[:, 0] = u[:, 1]
        u_perp[:, 1] = -u[:, 0]
        # change of angle
        d_phi = np.arccos(np.sum(u[:-1] * u[1:], axis=1))
        d_phi[np.isnan(d_phi)] = 0
        # projection of acc on direction perpendicular to vel
        a_v = np.abs(np.sum(acc * u[:-1], axis=1))
        a_phi = np.abs(np.sum(acc * u_perp[:-1], axis=1))

        # correct the time:
        output[:, i, :2] = pos[1:-1]
        output[:, i, 2:4] = vel[:-1]
        output[:, i, 4:6] = acc
        output[:, i, 6] = phi[:-1]
        output[:, i, 7] = a_v
        output[:, i, 8] = a_phi
        output[:, i, 9] = a_s
        output[:, i, 10] = d_phi
    return output


def get_IID(dat):
    '''
    returns <IID>(t) = inter individual distance TS
    INPUT: 
        dat.shape(time, N, 2) last 2 are x, y
    OUTPUT:
        result.shape(time)
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 3, 'len(dat.shape) != 3'
    time, N, _ = dat.shape
    result = np.empty(time)
    iuot = np.triu_indices(N, 1)    # indices of upper triangular matrix without diagonal
    for i in range(time):
        x = dat[i, :, 0]
        x1 = x.reshape(N, 1)
        x2 = x.reshape(1, N)
        xdiff = (x1-x2)[iuot].flatten()

        y = dat[i, :, 1]
        y1 = y.reshape(N, 1)
        y2 = y.reshape(1, N)
        ydiff = (y1-y2)[iuot].flatten()

        result[i] = np.sqrt(xdiff**2 + ydiff**2).mean()
    return result


def get_NND(dat):
    '''
    returns NND = nearest neighbor distance
    INPUT: 
        dat.shape(time, N, 2) last 2 are x, y
    OUTPUT:
        result.shape(time, N)
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 3, 'len(dat.shape) != 3'
    time, N, _ = dat.shape
    result = np.empty((time, N))
    iuot = np.triu_indices(N, 1)    # indices of upper triangular matrix without diagonal
    for i in range(time):
        x = dat[i, :, 0]
        x1 = x.reshape(N, 1)
        x2 = x.reshape(1, N)
        xdiff = (x1-x2)

        y = dat[i, :, 1]
        y1 = y.reshape(N, 1)
        y2 = y.reshape(1, N)
        ydiff = (y1-y2)

        dist = np.sqrt(xdiff**2 + ydiff**2)
        np.fill_diagonal(dist, dist.max() + 1)
        result[i] = dist.min(axis=0)
    return result


def get_ConvexHull_Area(dat):
    '''
    returns <IID>(t) = inter individual distance TS
    INPUT: 
        dat.shape(time, N, 2) last 2 are x, y
    OUTPUT:
        result.shape(time)
    '''
    assert dat.shape[-1] == 2, 'dat.shape[-1] != 2'
    assert len(dat.shape) == 3, 'len(dat.shape) != 3'
    time, N, _ = dat.shape
    result = np.empty(time)
    for i in range(time):
        result[i] = ConvexHull(dat[i]).volume
    return result
