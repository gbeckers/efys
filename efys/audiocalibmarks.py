import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
import uts
from darr import DataDir, create_datadir
from pathlib import Path

from . import edf
from . import openbci

def normalize(a, axis=0):
    a -= a.mean(axis=axis)
    a /= a.std()
    return a


def find_calibmarks(snd, calibmark, searchduration=30., recordedasbit=False, bitthreshold=0.005,
                    correct_ones=None):
    """Finds calibration stimuli at beginning and end of a longer sound (usually
    a recording of stimulus playback).

    Parameters
    ----------
    snd: UniformTimeSeries
        Signal in which the calibmarks should be detected
    calibmark: UniformTimeSeries
        Calibmark stimulus
    searchduration: float
        Duration of episode in beginning and end of `snd` in which calibmark will be searched for.
    recordedasbit: bool
        Is snd a bitsnd (i.e. just 0 and 1 values)?
    bitthreshold: float
        Threshold above which sample values of calibmark will be considered 1. Rest is 0.
    correct_ones: int | None
        If not None, indicates how long a sequence of ones should be in order to set them to 0.

    Returns
    -------
    (starttime1, starttime2)

    """

    if calibmark.fs != snd.fs:
        calibmark = uts.resample(calibmark, snd.fs / calibmark.fs)
    # first calibmark
    searchnframes = int(searchduration * snd.fs)
    calibsamples = calibmark.samples.get(False).astype('float64')
    if recordedasbit:
        calibsamples = (calibsamples > bitthreshold).astype('float64')
    target1 = snd.samples[:searchnframes].astype('float64')
    if correct_ones is not None:
        a = np.correlate(target1, np.ones(correct_ones), 'same')
        target1[a == correct_ones] = 0.
    # plt.figure()
    # plt.plot(snd[:searchnframes].samplingtimes(),target1)
    #target1 = normalize(target1)
    cc = np.correlate(target1, calibsamples, mode='valid')
    r1 = cc.argmax()
    # second calibmark
    target2 = snd.samples[-searchnframes:].astype('float64')
    if correct_ones is not None:
        a = np.correlate(target2, np.ones(correct_ones), 'same')
        target2[a == correct_ones] = 0.
    # plt.figure()
    # plt.plot(snd[-searchnframes:].samplingtimes(),target2)
    # #target2 = normalize(target2)
    cc = np.correlate(target2, calibsamples, mode='valid')
    # plt.figure()
    # plt.plot(cc)
    r2 = cc.argmax() + snd.ntimesamples - searchnframes
    return snd.index_to_time((r1, r2))


def convert_stimulustable(audiostimulustable, starttimefirst, starttimelast, newfs):
    """Convert audio stimulus table in a recording event table, if you know the
    recording starttimes of the first and the last sound event in the table.

    This can be used if there are no calibmarks to be automatically found, but you do
    have an idea of where they are.

    """

    st = audiostimulustable.copy()
    factor = (starttimelast - starttimefirst) / (st.iloc[-1]['starttime'] - st.iloc[0]['starttime'])
    offset = starttimefirst
    st['starttime'] = offset + factor * st['starttime']
    st['endtime'] = offset + factor * st['endtime']
    st['startframe'] = np.round(st['starttime'] * newfs).astype('int64')
    st['endframe'] = np.round(st['endtime'] * newfs).astype('int64')
    params = {'offset': offset, 'scalingfactor': factor}
    return st, params


# TODO settings euts fig and plots
def create_recordingstimulustable(recordedsnd, audiostimulustable, snd, recordedasbit=False,
                                  searchduration=30., bitthreshold=0.005, checkcalibmarks=False,
                                  correct_ones=None):
    """Create a stimulus timing table of recording based on calibration sounds.

    Parameters
    ----------
    recordedsnd
    audiostimulustable
    snd
    recordedasbit
    searchduration
    bitthreshold
    checkcalibmarks

    Returns
    -------
    st, params, (fig1, fig2)
    """

    st = audiostimulustable.copy()
    for i, pos in zip((0,-1),('first', 'last')):
        if not st.iloc[i]['snd'] in ('calibmark'):
            raise ValueError(f'{pos} row of playback stimulus table does not '
                             f'contain calibmark')
    calibmark = snd[st.iloc[0]['starttime']:st.iloc[0]['endtime']]
    t1,t2 = find_calibmarks(snd=recordedsnd, calibmark=calibmark,
                            searchduration=searchduration, recordedasbit=recordedasbit,
                            bitthreshold=bitthreshold, correct_ones=correct_ones)
    factor = (t2 - t1) / (st.iloc[-1]['starttime'] - st.iloc[0]['starttime'])
    offset = t1
    st['starttime'] = offset + factor * st['starttime']
    st['endtime'] = offset + factor * st['endtime']
    st['startframe'] = np.round(st['starttime'] * recordedsnd.fs).astype('int64')
    st['endframe'] = np.round(st['endtime'] * recordedsnd.fs).astype('int64')
    params = {'offset': offset, 'scalingfactor': factor}
    fig1 = fig2 = None
    if checkcalibmarks:
        cmdur = st.iloc[0]['endtime'] - st.iloc[0]['starttime']
        margin = int(round(cmdur*0.05*recordedsnd.fs))
        margin = min(margin, recordedsnd.ntimesamples-st.iloc[-1]['endframe'])
        detaildur = cmdur / 5
        detaillen = int(round(detaildur*recordedsnd.fs))
        detailmargin = detaillen // 10
        c1 = recordedsnd[st.iloc[0]['startframe'] - margin:st.iloc[0]['endframe'] + margin]
        c2 = recordedsnd[st.iloc[-1]['startframe'] - margin:st.iloc[-1]['endframe'] + margin]
        c1sel = recordedsnd[st.iloc[0]['startframe'] - detailmargin:st.iloc[0]['startframe'] + detaillen]
        c2sel = recordedsnd[st.iloc[-1]['startframe'] - detailmargin:st.iloc[-1]['startframe'] + detaillen]
        fig1 = plt.figure(figsize=(14,6))
        plt.subplot(4, 1, 1)
        plt.plot(c1.samplingtimes(), c1.samples[:])
        plt.title("Calibmarks")
        plt.subplot(4, 1, 2)
        plt.plot(c1sel.samplingtimes(), c1sel.samples[:])
        plt.subplot(4, 1, 3)
        plt.plot(c2.samplingtimes(), c2.samples[:])
        plt.subplot(4, 1, 4)
        plt.plot(c2sel.samplingtimes(), c2sel.samples[:])
        plt.xlabel('Recording time (s)')
        euts = recordedsnd.get_epochs(duration=0.3, starttimes=st['starttime']-0.1, origintime=0.1)
        fig2 = plt.figure(figsize=(14, 14))
        plt.imshow(euts.samples[:], cmap='hot', interpolation='nearest',
                   extent=[-0.1, 0.2, 0, len(st)], aspect='auto')
        plt.xlabel('Time re event (s)')
        plt.ylabel('Events')
        plt.title('Stimulus alignment')
    return st, params, (fig1, fig2)


def create_recordingstimulusinfobiosemi(edfpath, audiostimulustablepath, audiowavpath, outputpath=None,
                                        searchduration=30., bitthreshold=0.005):
    """Creates information on sound stimulus occurrence in recording.

    The information is generated based on a trace of the audio playback, a provided stimulus table
    and an playback audio file. The information is saved in several files in `outputpath`.

    Parameters
    ----------
    edfpath: path
    audiostimulustablepath: path
      Path to csv file with platback stimulus info
    audiowavpath
    outputpath
    searchduration
    bitthreshold

    Returns
    -------
    Pandas DataFrame recording stimulus table

    """
    overwrite = True
    audiochannel = 'Status'
    recordedasbit = True
    checkcalibmarks = True

    edfpath = Path(edfpath)
    if outputpath is None:
        outputpath = edfpath.with_name(f"{edfpath.with_suffix('').name}_stimulusinfo")
    else:
        outputpath = Path(outputpath)
    pst = pd.read_csv(audiostimulustablepath)
    bitsnd = edf.load_edfasumcts(str(edfpath), channels=[audiochannel], dtype='int16')[:,0]
    fs, data = wavfile.read(filename=str(audiowavpath))
    playbacksnd = uts.UniformTimeSeries(samples=data, fs=float(fs))
    st, params, (fig1, fig2) = create_recordingstimulustable(recordedsnd=bitsnd,
                                                             audiostimulustable=pst,
                                                             snd=playbacksnd,
                                                             recordedasbit=recordedasbit,
                                                             searchduration=searchduration,
                                                             bitthreshold=bitthreshold,
                                                             checkcalibmarks=checkcalibmarks)
    if not outputpath.exists():
        dd = create_datadir(outputpath)
    else:
        dd = DataDir(outputpath)
    dd.write_jsondict('timealignmentparameters.json',params, overwrite=overwrite)
    st.to_csv(dd.path/'recordingstimulustable.csv', index=False)
    pst.to_csv(dd.path / 'playbackstimulustable.csv', index=False)
    eventdict, eventtable = stimulustabletoevents(st)
    dd.write_jsondict('mne_eventdict.json', eventdict, overwrite=overwrite)
    eventtable.to_csv(dd.path / 'mne_eventtable.csv', index=False, header=False, sep='\t')
    fig1.savefig(dd.path / 'calibmarks.png', dpi=300)
    fig2.savefig(dd.path / 'snd_epochs.png', dpi=300)
    return st

def create_recordingstimulusinfoopenbci(datafilepath, audiostimulustablepath,
                                        audiowavpath, outputpath=None, searchduration=30.,
                                        bitthreshold=0.005):
    """Creates information on sound stimulus occurrence in recording.

    The information is generated based on a trace of the audio playback, a provided stimulus table
    and an playback audio file. The information is saved in several files in `outputpath`.

    Parameters
    ----------
    edfpath: path
    audiostimulustablepath: path
      Path to csv file with platback stimulus info
    audiowavpath
    outputpath
    searchduration
    bitthreshold

    Returns
    -------
    Pandas DataFrame recording stimulus table

    """
    overwrite = True
    datafilepath = Path(datafilepath)
    with open(datafilepath, 'r') as f:
        firstline = f.readline()
        if "OpenBCI Raw EEG Data" in firstline:
            audiochannel = 'other03'
            bitsnd = openbci.load_openbcidata(filepath=datafilepath)[:, audiochannel]
        else:
            audiochannel = 'accel_1'
            sampleindex, eeg, accel = openbci.load_thinkpulsedata(filepath=datafilepath)
            bitsnd = accel[:,[audiochannel]]
            bitsnd.samples.array[bitsnd.samples.array == 256] = 1
    recordedasbit = True
    checkcalibmarks = True
    if outputpath is None:
        outputpath = datafilepath.with_name(f"{datafilepath.with_suffix('').name}_stimulusinfo")
    else:
        outputpath = Path(outputpath)
    pst = pd.read_csv(audiostimulustablepath)

    fs, data = wavfile.read(filename=str(audiowavpath))
    playbacksnd = uts.UniformTimeSeries(samples=data, fs=float(fs))
    st, params, (fig1, fig2) = create_recordingstimulustable(recordedsnd=bitsnd,
                                                             audiostimulustable=pst,
                                                             snd=playbacksnd,
                                                             recordedasbit=recordedasbit,
                                                             searchduration=searchduration,
                                                             bitthreshold=bitthreshold,
                                                             checkcalibmarks=checkcalibmarks)
    if not outputpath.exists():
        dd = create_datadir(outputpath)
    else:
        dd = DataDir(outputpath)
    dd.write_jsondict('timealignmentparameters.json',params, overwrite=overwrite)
    st.to_csv(dd.path/'recordingstimulustable.csv', index=False)
    pst.to_csv(dd.path / 'playbackstimulustable.csv', index=False)
    eventdict, eventtable = stimulustabletoevents(st)
    dd.write_jsondict('mne_eventdict.json', eventdict, overwrite=overwrite)
    eventtable.to_csv(dd.path / 'mne_eventtable.csv', index=False, header=False, sep='\t')
    fig1.savefig(dd.path / 'calibmarks.png', dpi=300)
    fig2.savefig(dd.path / 'snd_epochs.png', dpi=300)
    return st



def stimulustabletoevents(st):
    """Converts a stimulus table (Pandas DataFrame) to an event table and event dictionary for MNE"""
    # TODO, this should allow for hierarchical events
    eventlabels = st['snd'].unique()
    eventdict = {l: i for i,l in enumerate(eventlabels)}
    eventtable = st[['startframe','endframe']].copy()
    eventtable['endframe'] = 0
    eventtable['events'] = st.apply(lambda x: eventdict[x['snd']], axis=1)
    return eventdict, eventtable