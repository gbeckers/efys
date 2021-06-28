import sys
import numpy as np
import uts
from pathlib import Path


def _kernelduration_to_ntaps(kernelduration, fs):
    return uts.core.utils.round_nearestodd(kernelduration * fs)

def _decfactor(newfs, fs):
    return int(np.round(fs / float(newfs)))

def _check_pathexists(path, overwrite):
    if Path(path).exists() and not overwrite:
        raise IOError(f'Path "{path}" already exists, use `overwrite` parameter to overwrite')


def create_lplfp(raw, path='lfp_lp.darr', freq=200., width=50., kernelduration=0.05,
                 window="kaiser", mode='same', dtype=np.float32, newfs=1000.,
                 overwrite=False, reportprogress=True, chunksize=1024 * 75,
                 threads=None):

    decf = _decfactor(newfs, raw.fs)
    ntaps = _kernelduration_to_ntaps(kernelduration=kernelduration,
                                     fs=raw.fs)
    attrs = {'lpntaps': ntaps, 'lpfs': raw.fs, 'lpfreq': freq,
             'lpwidth': width,
             'lpwindow': window, 'lpmode': mode, 'decimationfactor': decf}
    _check_pathexists(path=path, overwrite=overwrite)
    with uts.cacheh5_itered(raw,
                          uts.ilp(raw, freq=freq, width=width, ntaps=ntaps,
                                  window=window, mode=mode,
                                  dtype=np.float64,
                                  reportprogress=reportprogress,
                                  threads=threads,
                                  chunksize=chunksize),
                          dtype=np.float64, attrs=attrs, channelaxis=1) as lplfp:
        with uts.cacheh5(uts.iter_decimate(lplfp, decf)) as ar:
            return uts.savedarr(path=path,s=ar, dtype=dtype, axisorder=['time','channel'], overwrite=overwrite)


def create_bplfp(raw, path='lfp_bp.darr', freqs=(1.0, 200.), widths=(0.5, 50.),
                 kerneldurations=(4.0, 0.05), window="kaiser",
                 mode='same', newfs=1000., dtype=np.float32,
                 overwrite=False, reportprogress=True, chunksize=1024 * 75,
                 threads=None):
    decf = _decfactor(newfs, raw.fs)
    lpntaps = _kernelduration_to_ntaps(kernelduration=kerneldurations[1],
                                       fs=raw.fs)
    attrs = {'lpntaps': lpntaps, 'lpfs': raw.fs, 'lpfreq': freqs[1],
             'lpwidth': widths[1], 'lpwindow': window, 'lpmode': mode,
             'decimationfactor': decf}
    _check_pathexists(path=path, overwrite=overwrite)
    if reportprogress:
        print("starting lowpass")
    with uts.cacheh5_itered(raw, uts.ilp(raw, freq=freqs[1],
                                        width=widths[1],
                                       ntaps=lpntaps,
                                       window=window, mode=mode,
                                       dtype=np.float64,
                                       reportprogress=reportprogress,
                                       threads=threads,
                                       chunksize=chunksize),
                          dtype=np.float64, attrs=attrs, channelaxis=1) as lplfp:
        if reportprogress:
            print("starting decimation")
        sys.stdout.flush()
        with uts.cacheh5_itered(lplfp, uts.iter_decimate(lplfp, decf,
                                                     reportprogress=reportprogress),
                              ntimesamples=lplfp.ntimesamples // decf,
                              fs=float(raw.fs / decf),
                              attrs=attrs) as reslplfp:
            hpntaps = _kernelduration_to_ntaps(kernelduration=kerneldurations[0], fs=reslplfp.fs)
            attrs['hpntaps'] = hpntaps
            attrs['hpfs'] = reslplfp.fs
            attrs['hpfreq'] = freqs[0]
            attrs['hpwidth'] = widths[0]
            attrs['hpwindow'] = window
            attrs['hpmode'] = mode
            if reportprogress:
                print("starting high-pass")
            sys.stdout.flush()
            with uts.cacheh5_itered(reslplfp, uts.ihp(reslplfp, freq=freqs[0], width=widths[0],
                                                      threads=threads, ntaps=hpntaps, window=window,
                                                      mode=mode, reportprogress=reportprogress,
                                                      dtype=np.float64),
                                    dtype=np.float64, attrs=attrs, channelaxis=1) as bplfp:
                return uts.savedarr(path=path, s=bplfp, dtype=dtype, axisorder=['time','channel'], overwrite=overwrite)


def create_amua(raw, path='amua.darr', hpfreq=350., width=50.,
                kernelduration=0.05, window="kaiser",
                mode='same', newfs=1000., dtype=np.float32,
                overwrite=False, reportprogress=True, threads=None,
                hfmeancorrect=False):

        decf = _decfactor(newfs, raw.fs)
        ntaps = _kernelduration_to_ntaps(kernelduration=kernelduration,
                                         fs=raw.fs)
        attrs = {'ntaps': ntaps, 'rawfs': raw.fs, 'hpfreq': hpfreq,
                 'width': width,
                 'window': window, 'mode': mode,
                 'decimationfactor': decf,
                 'hfmeancorrected': int(hfmeancorrect)}
        _check_pathexists(path=path, overwrite=overwrite)
        if reportprogress:
            print('starting high-pass')
            sys.stdout.flush()
        with uts.cacheh5_itered(raw, uts.iter_map(uts.abs, uts.ihp(raw,
                                                                  freq=hpfreq, width=width,
                                           ntaps=ntaps,
                                           window=window, mode=mode,
                                           dtype=np.float64,
                                           reportprogress=reportprogress,
                                           threads=threads)),
                              dtype=np.float64, attrs=attrs, channelaxis=1) as absspikes:
            if reportprogress:
                print('starting decimating and saving')
                sys.stdout.flush()
            with uts.cacheh5(uts.iter_decimate(absspikes, decf)) as ar:
                return uts.savedarr(path=path, s=ar, dtype=dtype, axisorder=['time','channel'], overwrite=overwrite)

def car(s):
    s = s.transpose(['time','channel'])
    chmean = s.mean(s, 'channel')
    return uts.add(s, -chmean.samples[:].reshape(-1,1))

def create_esa(raw, path='amua.darr', hpfreq=350., hpwidth=50., hpkernelduration=0.05,
               lpfreq=30., lpwidth=5., lpkernelduration=0.1,
                window="kaiser", mode='same', newfs=1000., dtype=np.float32,
                overwrite=False, reportprogress=True, threads=None):

    """ESA was obtained by first digitally re-referencing the raw neural signals
        with common average reference (CAR) and high-pass filtering with 1st-order
        Butterworth filter at 300 Hz. The filtered signals were then full-wave rectified,
        low-pass filtered with 1st-order Butterworth filter at 12 Hz and downsampled to 1 kHz.
        All the filtering processes were performed in both forward and backward directions. """

    decf = _decfactor(newfs, raw.fs)
    hpntaps = _kernelduration_to_ntaps(kernelduration=hpkernelduration,
                                     fs=raw.fs)
    lpntaps = _kernelduration_to_ntaps(kernelduration=lpkernelduration,
                                       fs=raw.fs)
    attrs = {'hpntaps': hpntaps, 'lpntaps': lpntaps, 'rawfs': raw.fs,
             'hpfreq': hpfreq, 'lpfreq': lpfreq,
             'hpwidth': hpwidth, 'lpwidth': lpwidth,
             'window': window, 'mode': mode,
             'decimationfactor': decf}
    _check_pathexists(path=path, overwrite=overwrite)
    if reportprogress:
        print('starting high-pass')
        sys.stdout.flush()
    with uts.cacheh5_itered(raw, uts.iter_map(uts.abs, uts.ihp(raw,
                                                               freq=hpfreq, width=hpwidth,
                                                               ntaps=lpntaps,
                                                               window=window, mode=mode,
                                                               dtype=np.float64,
                                                               reportprogress=reportprogress,
                                                               threads=threads)),
                            dtype=np.float64, attrs=attrs, channelaxis=1) as absspikes:
        with uts.cacheh5_itered(absspikes, uts.ilp(absspikes, freq=lpfreq, width=lpwidth,
                                                      threads=threads, ntaps=hpntaps, window=window,
                                                      mode=mode, reportprogress=reportprogress,
                                                      dtype=np.float64),
                                    dtype=np.float64, attrs=attrs, channelaxis=1) as esa:

            if reportprogress:
                print('starting decimating and saving')
                sys.stdout.flush()
            with uts.cacheh5(uts.iter_decimate(esa, decf)) as ar:
                return uts.savedarr(path=path, s=ar, dtype=dtype, axisorder=['time','channel'], overwrite=overwrite)







