"""This module implements its own filtering functions (based on uts) as well as
filtering functions based on MNE. The advantage of the latter is that it is a
citable standard, and that it chooses sensible defaults. The disadvantage is
though that under the hood it has to read whole channels into RAM memory at 64 bits.
In case of long recordings and or high sampling rates, this may become problematic
depending on your hardware.

"""

import sys
import numpy as np
import scipy
import uts
import mne
import warnings
import mne.filter
import io
from contextlib import redirect_stdout
from pathlib import Path


def create_filter(sfreq, l_freq, h_freq, filter_length='auto', l_trans_bandwidth='auto',
                  h_trans_bandwidth='auto', method='fir', iir_params=None, phase='zero',
                  fir_window='hamming', fir_design='firwin'):
    """Wrapper around MNE create filter function.

    We do this, because we want to capture the MNE stdout text on filter info, and also we want
    a dictionary on filter design parameters. This info should be saved with the filtered signals
    for reproducibility.

    Returns
    -------
    (fdata, filtparams)
        `fdata` is the the filter data that MNE returns, i.e. an array when fir and a dictionary when iir.
        `filtparams` is a JSON serializable dictionary with all filter design info, to be saved for reproducibility.

    """
    with io.StringIO() as buf, redirect_stdout(buf):
        fdata = mne.filter.create_filter(data=None, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                          filter_length=filter_length, l_trans_bandwidth=l_trans_bandwidth,
                                          h_trans_bandwidth=h_trans_bandwidth, method=method, iir_params=iir_params,
                                          phase=phase, fir_window=fir_window, fir_design=fir_design, verbose=True)
        stdoutputtext = buf.getvalue()
    if method == 'iir':
        fdata['padlen'] = int(fdata['padlen']) # otherwise not JSON serializable
    filtparams = {
        'srfeq': sfreq,
        'l_freq': l_freq,
        'h_freq': h_freq,
        'filter_length': filter_length,
        'l_trans_bandwidth': l_trans_bandwidth,
        'h_trans_bandwidth': h_trans_bandwidth,
        'method': method,
        'iir_params': iir_params,
        'phase': phase,
        'fir_window': fir_window,
        'fir_design': fir_design,
        'fdata': fdata,
        'mneversion': mne.__version__,
        'numpyversion': np.__version__,
        'scipyversion': scipy.__version__,
        'pythonversion': sys.version,
        'mnestdout': stdoutputtext
    }
    if method == 'fir':
        filtparams['fdata'] = fdata.tolist()
    return fdata, filtparams

def _kernelduration_to_ntaps(kernelduration, fs):
    return uts.core.utils.round_nearestodd(kernelduration * fs)

def _decfactor(newfs, fs):
    if newfs is not None:
        return int(np.round(fs / float(newfs)))
    else:
        return None

def _check_pathexists(path, overwrite):
    if Path(path).exists() and not overwrite:
        raise IOError(f'Path "{path}" already exists, use `overwrite` parameter to overwrite')

def filtermne(s, outputpath, lfreq, hfreq, newfs=None, filterlength='auto', ltransbandwidth='auto',
              htransbandwidth='auto', njobs=1, method='fir', iirparams=None, phase='zero', firwindow='hamming',
              firdesign='firwin', pad='reflect_limited', verbose=None, print_filterinfo=False,
              dtype='float32', overwrite=False):
    """Filtering based on MNE `filter_data` function.

    This is the preferred way of filtering.

    Parameters
    ----------
    s
    outputpath
    lfreq
    hfreq
    newfs
    filterlength
    ltransbandwidth
    htransbandwidth
    njobs
    method
    iirparams
    phase
    firwindow
    firdesign
    pad
    verbose
    print_filterinfo

    Returns
    -------
    uts signal

    """
    outputpath = Path(outputpath)
    _check_pathexists(path=outputpath, overwrite=overwrite)
    fdata, filtparams = create_filter(sfreq=s.fs, l_freq=lfreq, h_freq=hfreq, filter_length=filterlength,
                                      l_trans_bandwidth=ltransbandwidth, h_trans_bandwidth=htransbandwidth,
                                      method=method, iir_params=iirparams, phase=phase, fir_window=firwindow,
                                      fir_design=firdesign)
    if newfs is not None:
        decf = _decfactor(newfs, s.fs)
    else:
        decf = None
    params = {
        'filtparams': filtparams,
        'pad': pad,
        'origfs': s.fs,
        'newfs': newfs,
        'decimatefactor': decf,
    }
    with uts.cachedarr(s, axisorder=('channel', 'time'),
                       report=True, keep=False, dtype='float64') as rawfilt: # mne works with float64 only
        data = rawfilt.samples.array
        with data._open_array() as (memmap, fd):
            mne.filter.filter_data(data=memmap, sfreq=s.fs, l_freq=None, h_freq=hfreq,
                        filter_length=filterlength, l_trans_bandwidth=ltransbandwidth,
                        h_trans_bandwidth=htransbandwidth, n_jobs=njobs,
                        method=method, iir_params=iirparams, copy=False, phase=phase,
                        fir_window=firwindow, fir_design=firdesign, pad=pad, verbose=verbose)
            if newfs is not None:
                rawfilt = uts.iter_decimate(rawfilt, decf, reportprogress=False)
        s = uts.savedarr(outputpath, rawfilt, axisorder=('time', 'channel'),
                         dtype=dtype, overwrite=True)
        s.datadir.write_jsondict('filterparams.json', params, overwrite=True)
        return s

def iter_firfilteruts(s, lfreq, hfreq, newfs=None, filterlength='auto',
                     ltransbandwidth='auto', htransbandwidth='auto', phase='zero',
                     firwindow='hamming', firdesign='firwin', dtype='float32',
                     reportprogress=True, mode='same', chunksize=1024*75, threads=None,
                     cachingthreshold=256., axisorder=('time','channel')):
    """Yields chunks of a filtered input signal.


    Parameters
    ----------
    s
    lfreq
    hfreq
    newfs
    filterlength
    ltransbandwidth
    htransbandwidth
    phase
    firwindow
    firdesign
    dtype
    reportprogress
    mode
    chunksize
    threads
    cachingthreshold
    axisorder: sequence of strings
        the order of the axes of the output chunks

    Returns
    -------
        Signal chunk generator

    """

    decf = _decfactor(newfs, s.fs)
    fdata, filtparams = create_filter(sfreq=s.fs, l_freq=lfreq, h_freq=hfreq, filter_length=filterlength,
                                      l_trans_bandwidth=ltransbandwidth, h_trans_bandwidth=htransbandwidth,
                                      method='fir', phase=phase, fir_window=firwindow,
                                      fir_design=firdesign)
    filtparams['mode'] = mode
    icframes = uts.iter_convolve(s, kernel=fdata, mode=mode, dtype='float64', reportprogress=reportprogress,
                                 threads=threads, chunksize=chunksize, axisorder=axisorder)
    if decf is None:
        for chunk in icframes:
            chunk.attrs.update({'filterparameters': filtparams})
            yield chunk
    else:
        filtparams['decimationfactor'] = decf
        filtparams['newfs'] = newfs
        filteredsize = s.mb * 8 / s.dtype.itemsize
        if filteredsize  >  cachingthreshold:
            with uts.cachedarr(icframes, dtype=dtype, keep=False, report=True) as s:
                for chunk in uts.iter_decimate(s, decf, chunksize=chunksize):
                    chunk.attrs.update({'filterparameters': filtparams})
                    yield chunk
        else:
            s = uts.concatenate_time(icframes)
            for chunk in uts.iter_decimate(s, decf, chunksize=chunksize):
                chunk.attrs.update({'filterparameters': filtparams})
                yield chunk


def firfilteruts(s, outputpath, lfreq, hfreq, newfs=None, filterlength='auto', ltransbandwidth='auto',
                 htransbandwidth='auto', phase='zero', firwindow='hamming', firdesign='firwin', dtype='float32',
                 overwrite=False, reportprogress=True, mode='same', chunksize=1024*75, threads=None):
    decf = _decfactor(newfs, s.fs)
    fdata, filtparams = create_filter(sfreq=s.fs, l_freq=lfreq, h_freq=hfreq, filter_length=filterlength,
                                      l_trans_bandwidth=ltransbandwidth, h_trans_bandwidth=htransbandwidth,
                                      method='fir', phase=phase, fir_window=firwindow,
                                      fir_design=firdesign)
    filtparams['mode'] = mode
    _check_pathexists(path=outputpath, overwrite=overwrite)
    sfilt = uts.convolve(s, kernel=fdata, mode=mode, dtype='float64',
                         reportprogress=reportprogress,
                         threads=threads, chunksize=chunksize)
    if decf is not None:
        filtparams['decimationfactor'] = decf
        filtparams['newfs'] = newfs
        sfilt = uts.decimate(sfilt, decf)
    sfilt.attrs.update({'filterparameters': filtparams})
    return sfilt


def create_lplfp(s, path='lfplp', freq=200., htransbandwidth='auto', filterlength='auto',
                 firwindow='hamming', phase='zero', firdesign='firwin', mode='same',
                 dtype='float32', newfs=1000., overwrite=False, reportprogress=True,
                 chunksize=1024*75, threads=None):
    s = iter_firfilteruts(s, lfreq=None, hfreq=freq, newfs=newfs,
                           filterlength=filterlength,
                           htransbandwidth=htransbandwidth, phase=phase,
                           firwindow=firwindow, firdesign=firdesign,
                           dtype=dtype, reportprogress=reportprogress,
                           mode=mode, chunksize=chunksize, threads=threads,
                           axisorder=('time','channel'))
    return uts.savedarr(path, s, axisorder=['time','channel'], overwrite=overwrite)

def create_bplfp(s, path='bplfp', freqs=(0.5, 200.), newfs=1000., ltransbandwidth='auto', htransbandwidth='auto',
                 filterlength='auto', firwindow='hamming', phase='zero', firdesign='firwin',
                 mode='same', dtype=np.float32, overwrite=False, reportprogress=True,
                 chunksize=1024*75, threads=None):
    _check_pathexists(path=path, overwrite=overwrite)
    filterparams = {}
    with uts.io.utils.tempdir(dir=None, keep=False, report=True) as tempdirname:
        hp = iter_firfilteruts(s, lfreq=None,
                               hfreq=freqs[1], newfs=newfs, filterlength=filterlength,
                            htransbandwidth=htransbandwidth, phase=phase, firwindow=firwindow, firdesign=firdesign,
                            dtype=dtype, reportprogress=reportprogress, mode=mode,
                            chunksize=chunksize, threads=threads,
                            axisorder=('time','channel'))
        lp = uts.savedarr(tempdirname, hp, overwrite=True)
        filterparams['lp'] = lp.attrs['filterparameters']
        bp = iter_firfilteruts(lp, lfreq=freqs[0], hfreq=None,
                           newfs=None, filterlength=filterlength,
                        ltransbandwidth=ltransbandwidth, phase=phase, firwindow=firwindow, firdesign=firdesign,
                        dtype=dtype, reportprogress=reportprogress, mode=mode,
                        chunksize=chunksize, threads=threads,
                               axisorder=('time','channel'))
        s = uts.savedarr(path, bp, overwrite=True)
        filterparams['hp'] = s.metadata['filterparameters']
        s.datadir.update_jsondict('metadata.json',{'filterparameters':
                                                   filterparams})
        s.datadir.write_jsondict('filterparams.json',
                                 filterparams, overwrite=True)
        return s

def create_amua(s, path='amua', freq=350., transbandwidth='auto',
                filterlength='auto', firwindow='hamming', phase='zero',
                firdesign='firwin',
                mode='same', newfs=1000., dtype=np.float32,
                overwrite=False, reportprogress=True, threads=None,
                chunksize=1024*75, carfilter=True):

    _check_pathexists(path=path, overwrite=overwrite)
    decf = _decfactor(newfs, s.fs)
    if reportprogress:
        print('starting high-pass')
        sys.stdout.flush()
    hp = iter_firfilteruts(s, lfreq=freq, hfreq=None,
                           newfs=None, filterlength=filterlength,
                           ltransbandwidth=transbandwidth, phase=phase,
                           firwindow=firwindow, firdesign=firdesign,
                           dtype=dtype, reportprogress=reportprogress,
                           mode=mode, chunksize=chunksize, threads=threads,
                           axisorder=('time','channel'))
    with uts.cachedarr(hp, report=True, keep=False, dtype='float64', axisorder=['time','channel']) as sfilt:
        if reportprogress:
            if carfilter:
                s = ' car filtering and'
            else:
                s = ''
            print(f'starting{s} taking absolute')
        da = sfilt.samples.array
        with da.open():
            for i,j in da.iterindices(chunklen=chunksize):
                chunk = da._memmap[i:j]
                if carfilter:
                    chunk -= np.mean(chunk, axis=1, keepdims=True)
                da._memmap[i:j] = np.absolute(chunk)
        deciterframes = uts.iter_decimate(sfilt, decf, reportprogress=reportprogress)
        if reportprogress:
            print('starting decimating and saving amua signal')
            sys.stdout.flush()
        s = uts.savedarr(path=path, s=deciterframes, dtype=dtype,
                         axisorder=['time','channel'], overwrite=True)
        filterparams = s.metadata['filterparameters']
        filterparams['decimationfactor'] = decf
        filterparams['newfs'] = newfs
        s.datadir.update_jsondict('metadata.json',{'filterparameters':
                                                   filterparams})
        s.datadir.write_jsondict('filterparams.json',
                                 filterparams, overwrite=True)
        return s

def create_esa(s, path='esa', hpfreq=350., lpfreq=30., hptransbandwidth='auto',
               lptransbandwidth='auto', hpfilterlength='auto',
               lpfilterlength='auto', firwindow='hamming', phase='zero',
                firdesign='firwin', mode='same', newfs=1000., dtype=np.float32,
                overwrite=False, reportprogress=True, threads=None,
                chunksize=1024*75):

    _check_pathexists(path=path, overwrite=overwrite)
    if reportprogress:
        print('starting high-pass')
        sys.stdout.flush()
    hp = iter_firfilteruts(s, lfreq=hpfreq, hfreq=None,
                           newfs=None, filterlength=hpfilterlength,
                           ltransbandwidth=hptransbandwidth, phase=phase,
                           firwindow=firwindow, firdesign=firdesign,
                           dtype=dtype, reportprogress=reportprogress,
                           mode=mode, chunksize=chunksize, threads=threads)
    filterparams = {}
    with uts.cachedarr(hp, report=True, keep=False, dtype='float64',
                       axisorder=['time','channel']) as sfilt:
        filterparams['hp'] = sfilt.metadata['filterparameters']
        if reportprogress:
            print(f'starting car filtering and taking absolute')
        da = sfilt.samples.array
        with da.open():
            for i,j in da.iterindices(chunklen=chunksize):
                chunk = da._memmap[i:j]
                chunk -= np.mean(chunk, axis=1, keepdims=True)
                da._memmap[i:j] = np.absolute(chunk)
        lp = iter_firfilteruts(sfilt, lfreq=None, hfreq=lpfreq,
                               newfs=newfs, filterlength=lpfilterlength,
                               htransbandwidth=lptransbandwidth, phase=phase,
                               firwindow=firwindow, firdesign=firdesign,
                               dtype=dtype, reportprogress=reportprogress,
                               mode=mode, chunksize=chunksize, threads=threads)
        if reportprogress:
            print('starting low-pass filtering, decimating and saving esa '
                  'signal')
            sys.stdout.flush()
        s = uts.savedarr(path=path, s=lp, dtype=dtype,
                         axisorder=['time','channel'], overwrite=True)
        filterparams['lp'] = s.metadata['filterparameters']
        s.datadir.update_jsondict('metadata.json',{'filterparameters':
                                                   filterparams})
        s.datadir.write_jsondict('filterparams.json',
                                 filterparams, overwrite=True)
        return s



#TODO remove dask dependence, use uts
def create_esamne(s, outputpath, overwrite=False):

    import dask

    """
    Drebitz, Eric, Bastian Schledde, Andreas K. Kreiter, and Detlef Wegener.
    “Optimizing the Yield of Multi-Unit Activity by Including the Entire Spiking
    Activity.” Frontiers in Neuroscience 13 (2019). https://doi.org/10.3389/fnins.2019.00083.

    ESA was obtained by first digitally re-referencing the raw neural signals
    with common average reference (CAR) and high-pass filtering with 1st-order
    Butterworth filter at 300 Hz. The filtered signals were then full-wave rectified,
    low-pass filtered with 1st-order Butterworth filter at 12 Hz and downsampled to 1 kHz.
    All the filtering processes were performed in both forward and backward directions. """

    newfs = 1000.
    oldfs = float(s.fs)
    outputpath = Path(outputpath)
    if outputpath.exists() and not overwrite:
        warnings.warn(f"'{outputpath}' exists and `overwrite` is False, skipping filtering ...")
        return
    decf = _decfactor(newfs, s.fs)
    iir_params_hp = dict(order=1, ftype='butter', output='sos')
    iir_params_hp = mne.filter.construct_iir_filter(iir_params_hp, 300., None, s.fs, 'high', return_copy=False)
    iir_params_lp = dict(order=1, ftype='butter', output='sos')
    iir_params_lp = mne.filter.construct_iir_filter(iir_params_lp, 100., None, s.fs, 'low', return_copy=False)
    with uts.cachedarr(s, axisorder=('channel', 'time'),
                       report=True, keep=False, dtype='float64') as rawfilt:  # mne works with float64 only
        data = rawfilt.samples.array
        if not outputpath.exists():
            outputpath.mkdir()
        infofile = outputpath / 'mnestdoutinfo.txt'
        if infofile.exists():
            infofile.unlink()
        mne.set_log_file(infofile)
        chmean = uts.mean(rawfilt, 'channel').samples[:]
        with rawfilt.samples.array.open(): # CAR filtering
            a = rawfilt.samples.array # darr array
            dara = dask.array.from_array(a, chunks=(rawfilt.nchannels, 512))
            dask.array.add(dara,chmean).store(a)
        with data._open_array() as (memmap, fd):
            mne.filter.filter_data(data=memmap, sfreq=s.fs, l_freq=None, h_freq=300., method='iir',
                        iir_params=iir_params_hp, copy=False)
            # take abs in place
            dara = dask.array.from_array(memmap, chunks=(rawfilt.nchannels, 512))
            dask.array.absolute(dara).store(memmap)
            mne.filter.filter_data(data=memmap, sfreq=s.fs, l_freq=100., h_freq=None, method='iir',
                        iir_params=iir_params_lp, copy=False)
            rawfilt = uts.iter_decimate(rawfilt, decf, reportprogress=False)
        s = uts.savedarr(outputpath, rawfilt, axisorder=('time', 'channel'),
                         dtype='float32', overwrite=True)
        iir_params_lp['sos'] = (iir_params_lp['sos']).tolist()
        iir_params_lp['padlen'] = int(iir_params_lp['padlen'])
        iir_params_hp['sos'] = (iir_params_hp['sos']).tolist()
        iir_params_hp['padlen'] = int(iir_params_hp['padlen'])
        filtparams = {
            'method': 'iir',
            'iir_params_hp': iir_params_hp,
            'iir_params_lp': iir_params_lp,
            'mneversion': mne.__version__,
            'utsversion': uts.__version__,
            'origfs': oldfs,
            'newfs': newfs,
            'decimatefactor': decf,
        }
        s.datadir.update_jsondict('metadata.json', {'filterparameters':
                                                        filtparams})
        s.datadir.write_jsondict('filterparams.json', filtparams, overwrite=True)
        mne.set_log_file(None)

        return s


