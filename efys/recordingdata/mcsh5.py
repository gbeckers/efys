"""Module to read data from MCS hdf5 files.

It maps the data directly to the MultiChannelUniformTimeSeries objects,
which is very efficient.

"""

# TODO: this file is entirely based on a 8x8 NN matrix electrode

import time
import datetime
import warnings
import tables as tb
import numpy as np
import uts
from uts.core.samplesource import H5pyArraySamples
from uts.core.geometry import RectangularLattice
import h5py
from contextlib import contextmanager

a8x8_rev5_electrode_to_twin_mpa32i = {  1: (2, 29),
                                        2: (2, 27),
                                        3: (2, 31),
                                        4: (2, 25),
                                        5: (2, 30),
                                        6: (2, 23),
                                        7: (2, 28),
                                        8: (2, 21),
                                        9: (2, 26),
                                       10: (2, 19),
                                       11: (2, 24),
                                       12: (1, 33),
                                       13: (2, 22),
                                       14: (1, 31),
                                       15: (2, 20),
                                       16: (1, 29),
                                       17: (1, 34),
                                       18: (1, 27),
                                       19: (1, 32),
                                       20: (1, 25),
                                       21: (1, 30),
                                       22: (1, 21),
                                       23: (1, 28),
                                       24: (1, 19),
                                       25: (1, 26),
                                       26: (1, 24),
                                       27: (1, 20),
                                       28: (1, 22),
                                       29: (1, 23),
                                       30: (2, 32),
                                       31: (2, 33),
                                       32: (2, 34),
                                       33: (2,  3),
                                       34: (2,  4),
                                       35: (2,  5),
                                       36: (1, 14),
                                       37: (1, 15),
                                       38: (1, 17),
                                       39: (1, 13),
                                       40: (1, 11),
                                       41: (1, 18),
                                       42: (1,  9),
                                       43: (1, 16),
                                       44: (1,  7),
                                       45: (1, 12),
                                       46: (1,  5),
                                       47: (1, 10),
                                       48: (1,  3),
                                       49: (1,  8),
                                       50: (2, 17),
                                       51: (1,  6),
                                       52: (2, 15),
                                       53: (1,  4),
                                       54: (2, 13),
                                       55: (2, 18),
                                       56: (2, 11),
                                       57: (2, 16),
                                       58: (2,  9),
                                       59: (2, 14),
                                       60: (2,  7),
                                       61: (2, 12),
                                       62: (2,  6),
                                       63: (2, 10),
                                       64: (2,  8)
                                       }

def get_nn8x8_geometry(dx=0.2):
    dy = 0.2
    nx = 8
    ny = 8
    order = [[5,13,21,29,37,45,53,61],
             [4,12,20,28,36,44,52,60],
             [6,14,22,30,38,46,54,62],
             [3,11,19,27,35,43,51,59],
             [7,15,23,31,39,47,55,63],
             [2,10,18,26,34,42,50,58],
             [8,16,24,32,40,48,56,64],
             [1,9,17,25,33,41,49,57]]
    channels = []
    for rowi, row in enumerate(order):
        for coli, elnr in enumerate(row):
            channels.append('electrode_%.2d' % elnr)

    return RectangularLattice(nx=nx, ny=ny, dx=dx, dy=dy,
                              pointnames=channels)


def get_acute_64_rev5_index():
    el_chan = np.array(
            [[key, (item[0] - 1) * 32 + (item[1] - 2)] for key, item in a8x8_rev5_electrode_to_twin_mpa32i.items()])
    return np.argsort(el_chan[:, 1])


class Mcsh5Samples(H5pyArraySamples):
    def __init__(self, h5array, axes, scale, adczero, readonly=True):
        warnings.simplefilter(action='ignore', category=tb.NaturalNameWarning)
        H5pyArraySamples.__init__(self, h5array, axes=axes,
                                      readonly=True)
        self._scale = float(scale)
        self._adczero = float(adczero)
        self._set_dtype('float64')

    def __getitem__(self, index):
        return (self.h5pyarray[index] - self._adczero) * self._scale


@contextmanager
def open_mcsh5stream(filename, recno=0, streamno=1):
    reclabel = 'Recording_{}'.format(recno)
    streamlabel = 'Stream_{}'.format(streamno)
    attrd = {}
    with  h5py.File(filename, 'r') as f:
        adczero = f['Data'][reclabel]['AnalogStream'][streamlabel]['InfoChannel'][0]['ADZero']
        exponent = f['Data'][reclabel]['AnalogStream'][streamlabel]['InfoChannel'][0]['Exponent']
        conversionfactor = f['Data'][reclabel]['AnalogStream'][streamlabel]['InfoChannel'][0]['ConversionFactor']
        tick  =  f['Data'][reclabel]['AnalogStream'][streamlabel]['InfoChannel'][0]['Tick']
        fs = 1e6/float(tick)
        scale = conversionfactor * (10 ** exponent.astype(np.float64))
        date_in_clr_ticks = f['Data'].attrs['DateInTicks']
        date = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds=int(date_in_clr_ticks) / 10)
        starttime = time.mktime(date.timetuple())
        axes = {'channel': 0, 'time': 1}
        attrd = {}
        attrd.update(f['Data'].attrs)
        attrd.update(f['Data'][reclabel].attrs)
        attrd.update(f['Data'][reclabel]['AnalogStream'].attrs)
        attrd.update(f['Data'][reclabel]['AnalogStream'][streamlabel].attrs)
        attrd.update(f['Data'][reclabel]['AnalogStream'][streamlabel]['ChannelDataTimeStamps'].attrs)
        attrd.update(f['Data'][reclabel]['AnalogStream'][streamlabel]['InfoChannel'].attrs)
        arnode = f['Data'][reclabel]['AnalogStream'][streamlabel]['ChannelData']
        samples = Mcsh5Samples(h5array=arnode, axes=axes, scale=scale, adczero=adczero)
        yield uts.MultiChannelUniformTimeSeries(samples=samples, fs=fs, channelaxis=0,
                                                copy=False,
                                                starttime=starttime,
                                                attrs=attrd)


@contextmanager
def open_nn8x8rev5recording(filename):
    with open_mcsh5stream(filename,0,0) as raw, open_mcsh5stream(filename,0,1) as snd:
        g = get_nn8x8_geometry()
        index = get_acute_64_rev5_index()
        channelnames = np.array(sorted(g.pointnames.flatten()))[index].tolist()
        raw.set_geometry(g,channelnames)
        yield raw, snd

# def load_mcsh5stream(filename, start=None, end=None, recno=0, streamno=1):
#     with open_mcsh5stream(filename=filename, recno=recno, streamno=streamno) as s:
#         return s[start:end]
