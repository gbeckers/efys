import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import json
import uts
from pathlib import Path

from . import edf

def create_recordingstimulustable(edfpath, playbackstimulustable, playbackwavpath, searchduration=30.,
                                  checkcalibmarks=False):
    edfpath = Path(edfpath)
    st = pd.read_csv(playbackstimulustable)
    if not st.iloc[0]['snd'] in ('calibmark'):
        raise ValueError(f'first row of playback stimulus table does not '
                         f'contain calibmark')
    startframe = st.iloc[0]['startframe']
    endframe = st.iloc[0]['endframe']
    frames, fs = sf.read(playbackwavpath)
    bitsnd = edf.loadedfasumcts(str(edfpath), channels=['Status'], dtype='int16')[:,0]
    calibmark = uts.resample(uts.UniformTimeSeries(frames[startframe:endframe],
                                                   fs=float(fs)),
                             bitsnd.fs / fs)
    searchnframes = int(searchduration * bitsnd.fs)
    th = (calibmark.samples.get(False) > 0.005).astype('float64')
    target1 = bitsnd.samples[:searchnframes].astype('float64')
    cc1 = np.correlate(target1, th, mode='valid')
    offset = cc1.argmax()
    if not st.iloc[-1]['snd'] == 'calibmark':
        raise ValueError(f'last row of playback stimulus table does not '
                         f'contain calibmark')
    starttime = st.iloc[-1]['starttime']
    endtime = st.iloc[-1]['endtime']
    startframe = offset + int(round((0.99 * starttime * bitsnd.fs)))
    endframe = offset + int(round((1.01 * endtime * bitsnd.fs)))
    target2 = bitsnd.samples[startframe:endframe].astype('float64')# - 0.5
    cc2 = np.correlate(target2, th, mode='valid')
    s2 = cc2.argmax() + startframe
    factor = (bitsnd.index_to_time(s2) - bitsnd.index_to_time(offset)) \
             / (st.iloc[-1]['starttime'] - st.iloc[0]['starttime'])

    st['starttime'] = (offset / bitsnd.fs) + factor * st['starttime']
    st['endtime'] = (offset / bitsnd.fs) + factor * st['endtime']
    st['startframe'] = np.round(st['starttime'] * bitsnd.fs).astype('int64')
    st['endframe'] = np.round(st['endtime'] * bitsnd.fs).astype('int64')
    st.to_csv(f'{edfpath.with_suffix("")}_events.csv')
    with open(f'{edfpath.with_suffix("")}_calibration.json' ,mode='w+') as fp:
        json.dump({'offset': offset/bitsnd.fs, 'factor': factor}, fp)

    if checkcalibmarks:
        margin = 2000
        c1 = bitsnd[st.iloc[0]['startframe'] - margin:st.iloc[0]['endframe'] + margin]
        c2 = bitsnd[st.iloc[-1]['startframe'] - margin:st.iloc[-1]['endframe'] + margin]
        plt.figure(figsize=(14,6))
        plt.subplot(4, 1, 1)
        plt.plot(c1.samplingtimes(), c1.samples[:])
        plt.title("Calibmarks")
        plt.subplot(4, 1, 2)
        c1sel = c1[margin - 100:2 * margin]
        plt.plot(c1sel.samplingtimes(), c1sel.samples[:])
        plt.subplot(4, 1, 3)
        plt.plot(c2.samplingtimes(), c2.samples[:])
        plt.subplot(4, 1, 4)
        c2sel = c2[margin-100:2*margin]
        plt.plot(c2sel.samplingtimes(), c2sel.samples[:])
        plt.xlabel('Recording time (s)')
        plt.savefig(f'{edfpath.with_suffix("")}_calibmarkcheck.png', dpi=300)
        euts = bitsnd.get_epochs(duration=0.2, epochs=st.to_records())
        plt.figure(figsize=(14, 14))
        plt.imshow(euts.samples[:], cmap='hot', interpolation='nearest',
                   extent=[0, 0.2, 0, len(st)], aspect='auto')
        plt.xlabel('Time re event (s)')
        plt.ylabel('Events')
        plt.title('Stimulus alignment')
        plt.savefig(f'{edfpath.with_suffix("")}_stimulusalignment.png', dpi=300)
    return st

