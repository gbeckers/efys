import numpy as np
import pandas as pd
import uts

__all__ = ['load_thinkpulsedata']

def hexstrtomicrovolt(hexstr, minV=-4.5, maxV=4.5, gain=8):
    if pd.isna(hexstr):
        return hexstr
    bits = len(hexstr) * 4
    value = int(hexstr, 16)
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    value *= (maxV - minV) / (16.777216 * gain) # 16.777216 = 2**24 / 1e6
    return value

def strtobit(hexstr):
    if pd.isna(hexstr):
        return 0
    else:
        return int(hexstr)


# TODO, this function could be improved by just reading directly from text
def load_thinkpulsedata(filepath, fs=250., startdatetime='NaT'):
    eegcolumns =  ['channel_00','channel_01','channel_02','channel_03',
                   'channel_04', 'channel_05','channel_06','channel_07']
    accelcolumns = ['accel_0','accel_1','accel_2']
    sampleindexcolumn = ['sampleindex']
    columnnames = sampleindexcolumn + eegcolumns + accelcolumns
    dtype = {cn: str for cn in columnnames}
    df = pd.read_csv(filepath, skiprows=2, names=columnnames, dtype=dtype)
    for colname in eegcolumns:
        df[colname] = df[colname].apply(hexstrtomicrovolt)
    for colname in accelcolumns:
        df[colname] = df[colname].apply(lambda s: int(s, 16))
    df['sampleindex'] = df['sampleindex'].apply(lambda s: int(s, 16))
    eeg = uts.MultiChannelUniformTimeSeries(df[eegcolumns].to_numpy(),
                                          fs=float(fs), channelnames=eegcolumns,
                                          startdatetime=startdatetime)
    accel = uts.MultiChannelUniformTimeSeries(df[accelcolumns].to_numpy(),
                                          fs=float(fs), channelnames=accelcolumns,
                                          startdatetime=startdatetime)
    sampleindex = uts.UniformTimeSeries(df[sampleindexcolumn].values[:,0],
                                      fs=float(fs), startdatetime=startdatetime)
    return sampleindex, eeg, accel

#TODO improve reading header by regexp
# other channels, Analog and accel
def load_openbcidata(filepath):
    # % OpenBCI Raw EEG Data
    # % Number of channels = 8
    # % Sample Rate = 250 Hz
    # % Board = OpenBCI_GUI$BoardCytonSerial
    with open(filepath, 'r') as f:
        header = [f.readline() for i in range(4)]
    nchannels = int(header[1].split('=')[-1])
    fs = float(header[2].split('=')[-1][:-3])
    df = pd.read_csv(filepath, header=4, delimiter=',')
    eegchannels = [f' EXG Channel {i}' for i in range(nchannels)]
    eegsamples = df[eegchannels].to_numpy()
    eegchannelnames = [f'eeg{i:02d}' for i in range(nchannels)]
    otherchannels = [c for c in df.columns if "Other" in c]
    othersamples = df[otherchannels].to_numpy()
    otherchannelnames = [f'other{i:02d}' for i in range(len(otherchannels))]
    accelchannels = [' Accel Channel 0', ' Accel Channel 1', ' Accel Channel 2']
    accelsamples = df[accelchannels].to_numpy()
    accelchannelnames = ['accel0', 'accel1', 'accel2']
    analogchannels = [c for c in df.columns if "Analog" in c]
    analogsamples = df[analogchannels].to_numpy()
    analogchannelnames = [f'analog{i}' for i in range(len(analogchannels))]
    ts = df.iloc[0][' Timestamp (Formatted)']
    startdatetime = np.datetime64(ts[1:].replace(' ', 'T'))
    samples = np.concatenate([eegsamples, othersamples, accelsamples, analogsamples], axis=1)
    channelnames = eegchannelnames + otherchannelnames + accelchannelnames + analogchannelnames
    s = uts.MultiChannelUniformTimeSeries(samples,
                                          fs=fs, channelnames=channelnames,
                                          startdatetime=startdatetime)
    return s
