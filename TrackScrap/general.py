import numpy as np
import csv
import pandas as pd


def setDefault(x, val):
    if x is None:
        x = val
    return x


def RotateCcw(vecs, phi):
    '''
    INPUT:
        vecs.shape (N, 2)
        phi double
    OUTPUT:
        vecs_rot.shape(N, 2)
    '''
    vecs_rot = np.zeros(vecs.shape, dtype=float)
    vecs_rot[:, 0] = np.cos(phi) * vecs[:, 0] - np.sin(phi) * vecs[:, 1]
    vecs_rot[:, 1] = np.sin(phi) * vecs[:, 0] + np.cos(phi) * vecs[:, 1]
    return vecs_rot


def smooth2D(dat, std, min_periods=None, smotype=None):
    assert len(dat.shape) == 2, 'data has wrong dimension' 
    smodat = np.empty(dat.shape, dtype=float)
    for i, da in enumerate(dat.T):
        smodat[:, i] = smooth1D(da, std, min_periods=min_periods,
                                smotype=smotype)
    return smodat


def smooth1D(dat, std=None, window=None, min_periods=None, smotype=None):
    '''
    INPUT:
        dat.shape(T)
        std double
            - standard deviation of gaussian kernel used
            - if window=None: int(6*std) = window size
        window int
            number of datapoints in a moving window
        min_periods int
            - minimum # of datapoints in window
            - if less data than min_periods -> None
        smotype string
            up to now only 'gaussian' is implemented
    '''
    std = setDefault(std, 1)
    window = setDefault(window, int(np.round(6*std)))
    min_periods = setDefault(min_periods, 1)
    smotype = setDefault(smotype, 'gaussian')
    # use pandas to smooth
    smodat = pd.Series(dat)
    smodat = smodat.rolling(window=window, win_type=smotype,
                            center=True, min_periods=int(np.round(min_periods))
                           ).mean(std=std)
    return smodat


def clean_blocks(blocks):
    '''
    joins overlapping blocks
    '''
    in_block = np.zeros(blocks.max() + 1)
    for b in blocks:
        in_block[b[0]:b[1]+1] = 1
    out = find_blocks_large(in_block, 0.5, 1, 1)
    return out


def find_blocks_low(dat, mind, minNr, minBlock, noBlocks=None, lowEqual=None):
    '''
    find start- end end-time of blocks where
    dat<mind for at least "minNr" agents
    INPUT:
        dat.shape(time, N) OR (time)
            datability that ID is correct for "N" agents OR 1 agent
        minNr float
            # of agents for which the creterion must hold simultaneously
        minBlock int
            minium size of block-length
    '''
    if noBlocks is None:
        noBlocks = False
    if lowEqual is None:
        lowEqual = False 
    there = np.where(dat < mind)
    if lowEqual:
        there = np.where(dat <= mind)
    if len(dat.shape) > 1:  # dat.shape(time, N)
        Blowdats = np.zeros(dat.shape)
        Blowdats[there] = 1
        there = np.where(np.sum(Blowdats, axis=1) >= minNr)[0]
    else:                   # dat.shape(time)
        there = there[0]
        minBlock = 1
    if noBlocks:
        return there
    blocks = get_blocks(there, minBlock)
    return blocks

def find_blocks_large(dat, mind, minNr, minBlock, noBlocks=None):
    '''
    find start- end end-time of blocks where
    dat > mind for at least "minNr" agents
    INPUT:
        dat.shape(time, N) OR (time)
            datability that ID is correct for "N" agents OR 1 agent
        minNr float
            # of agents for which the creterion must hold simultaneously
        minBlock int
            minium size of block-length
    '''
    if noBlocks is None:
        noBlocks = False
    there = np.where(dat > mind)
    if len(dat.shape) > 1:  # dat.shape(time, N)
        Blowdats = np.zeros(dat.shape)
        Blowdats[there] = 1
        there = np.where(np.sum(Blowdats, axis=1) >= minNr)[0]
    else:                   # dat.shape(time)
        there = there[0]
        minBlock = 1
    if noBlocks:
        return there
    blocks = get_blocks(there, minBlock)
    return blocks

def get_blocks(there, minsize):
    '''
    return blocks(continously increasing values) in there
    and return the start and end-value of the blocks
    example: 
            in: there=[1, 2, 3, 10, 11, 12, 13, 14, 22, 25,26]
            out: blocks=[[1,3], [10, 14], [22, 22], [25, 26]]

        INPUT:
            there.shape(time)
        OUTPUT:
            blocks.shape(blocks, 2)
    '''
    if len(there) == 0:
        return []
    differs = np.diff(there)
    borders = np.where(differs>1)[0]
    sblocks = np.ones((len(borders) + 1, 1), dtype='int')  # start points of blocks
    eblocks = np.ones((len(borders) + 1, 1), dtype='int')  # end points of blocks
    sblocks[0] = there[0]  # first start point of block
    eblocks[-1] = there[-1]  # last end point of block
    for i in range(len(borders)):
        eblocks[i] = there[borders[i]]
        sblocks[i+1] = there[borders[i] + 1]
    blocks = np.hstack((sblocks, eblocks))
    blocklen = np.diff(blocks) + 1
    longenough = np.where(blocklen >= minsize)[0]
    blocks = blocks[longenough]
    blocklen = np.diff(blocks) + 1
    return blocks

def loadCsv(f_name, skiprows=None, skipcols=None):
    if skiprows is None:
        skiprows = 0
    if skipcols is None:
        skipcols = 0
    with open(f_name, 'r') as f_data:
        reader = csv.reader(f_data)
        i = 1
        data = []
        for row in reader:
            if i > skiprows:
                data.append(row)
            i += 1
        data = np.array(data)
    data[data=='NA'] = -111
    data = data.astype(float)
    data[data==-111] = np.nan 
    return data[:, skipcols:]
