import pandas as pd

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

def load_thinkpulsedata(filepath):
    df = pd.read_csv(filepath, skiprows=2,
                     names=['sampleindex','channel_00','channel_01','channel_02','channel_03','channel_04',
                           'channel_05','channel_06','channel_07','accel_0','accel_1','accel_2'])
    for colname in ('channel_00','channel_01','channel_02','channel_03','channel_04',
                    'channel_05','channel_06','channel_07','accel_0','accel_1','accel_2'):
        df[colname] = df[colname].apply(hexstrtomicrovolt)
    df['sampleindex'] = df['sampleindex'].apply(lambda s: int(s, 16))
    return df