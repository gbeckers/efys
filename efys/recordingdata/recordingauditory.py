"""Classes for recordings of auditory electrophysiology recordings.

"""

import warnings
import re
import numpy as np
import darr
import uts
import send2trash
import pandas as pd

import matplotlib.pyplot as plt
import efys
from contextlib import contextmanager
from pathlib import Path
from . import mcsh5
from efys.filtering import create_bplfp, create_lplfp, create_amua, create_esa
from efys.stimuli.auditory import AuditoryStimuli


class BaseEfysDir:
    """Base class for an efys recordingdata directory structure.

    It is essentially a directory with a json info file in it,
    containing information on the efys class the directory
    corresponds to, and the version of efys.

    """

    _infofilename = 'efys.json'

    def __init__(self, path):
        self._path = Path(path)
        if not self.path.exists():
            raise IOError(f'"{path}" does not exist')
        self._name = self._path.name
        self._datadir = darr.DataDir(self._path)
        self._infofilepath = self._datadir.path / self._infofilename
        self._check_infofile()

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def datadir(self):
        return self._datadir

    def _check_infofile(self):
        if not self._infofilepath.exists():
            self._update_infofile()

    def _update_infofile(self, classname=None):
        if classname is None:
            classname = self.__class__.__name__
        d = {'efysclass': classname,
             'efysversion': efys.__version__}
        self._datadir.write_jsondict(self._infofilepath,
                                     d=d, overwrite=True)


class Experiment(BaseEfysDir):

    """A directory that has subdirectories that represent recordingdata sessions
    from one experiment.

    Parameters
    ----------
    path: str or Path
        Directory path

    Returns
    -------
    Experiment object

    """

    def __init__(self, path):
        BaseEfysDir.__init__(self, path=path)

    def __str__(self):
        s = f"<Experiment: {self.path}>\n"
        for rn in self.recordingsessionnames:
            s += f"\t{rn}\n"
        return s

    __repr__ = __str__

    def __getitem__(self, key):
        return RecordingSession(self.path / key)

    def __iter__(self):
        for rn in self.recordingsessionnames:
            yield self[rn]

    @property
    def recordingsessionnames(self):
        rnames = sorted(path.name for path in self.path.glob('*') if path.is_dir())
        # only add if they do not fail
        validrnames = []
        for rname in rnames:
            try:
                r = self[rname]
                validrnames.append(rname)
            except Exception as e:
                warnings.warn(f"{rname} is a directory in {self._path} but "
                              f"holds no valid recordingdata data. Caught exception "
                              f"{str(e)}", ResourceWarning)
        return validrnames

    def _update_recordingclass(self, classname):
        for rs in self:
            for r in rs:
                r._update_recordingclass(classname)

    def search_recordings(self, regexp):
        d = {}
        for rs in self:
            recordings = rs.search_recordings(regexp=regexp)
            if len(recordings) > 0:
                d[rs.name] = {}
                for rname, r in recordings.items():
                    d[rs.name][rname] = r
        return d

    def iter_recordings(self, print_progress=False):
        """Iterates over recordings in recordinsessions in experiment

        Parameters
        ----------
        print_progress

        Returns
        -------
        iterates over (recordingsession, recordingdata) tuples

        """
        for rs in self:
            if print_progress:
                print(f'Recording session {rs.name}')
            for r in rs:
                if print_progress:
                    print(f'\tRecording {r.name}')
                yield rs, r


    def create_lplfp(self, freq=200., signalname='lplfp', threads=4, print_progress=True, overwrite=False):
        for r, rs in self.iter_recordings(print_progress=print_progress):
            r.create_lplfp(freq=freq, signalname=signalname, threads=threads, reportprogress=False, overwrite=overwrite)

    def create_amua(self, signalname='amua', threads=4, print_progress=True, overwrite=False):
        for r, rs in self.iter_recordings(print_progress=print_progress):
            r.create_amua(signalname=signalname, reportprogress=False, threads=threads, overwrite=overwrite)

    def create_filteredsignals(self, threads=4, overwrite=False, print_progress=False):
        for r, rs in self.iter_recordings(print_progress=print_progress):
            r.create_filteredsignals(threads=threads, overwrite=overwrite)



class RecordingSession(BaseEfysDir):
    """A directory that has subdirectories that represent recordings from one
    recordingdata session.

    Parameters
    ----------
    path: str or Path
        Directory path

    Returns
    -------
    RecordingSession object

    """

    def __init__(self, path):
        BaseEfysDir.__init__(self, path=path)

    def __str__(self):
        s = f"<RecordingSession: {self.path}>\n"
        for rn in self.recordingnames:
            s += f"\t{rn}\n"
        return s

    __repr__ = __str__

    def __getitem__(self, key):
        infofilepath = self.path / key / self._infofilename
        if infofilepath.exists():
            efysclass = self.datadir.read_jsondict(infofilepath)['efysclass']
            r =  recordingclasses[efysclass](self.path / key)
        else:

            r = RecordingAuditory(self.path / key)
        return r

    def __iter__(self):
        for rn in self.recordingnames:
            yield self[rn]

    @property
    def recordingnames(self):
        rnames = sorted(path.name for path in self.path.glob('*') if path.is_dir())
        # only add if they do not fail
        validrnames = []
        for rname in rnames:
            try:
                r = self[rname]
                validrnames.append(rname)
            except Exception as e:
                warnings.warn(f"{rname} is a directory in {self._path} but "
                              f"holds no valid recordingdata data. Caught exception "
                              f"{str(e)}", ResourceWarning)
                raise e
        return validrnames

    def _update_recordingclass(self, classname):
        for r in self:
            r._update_recordingclass(classname)

    def search_recordings(self, regexp):
        d = {}
        for name in self.recordingnames:
            if re.search(regexp, name) is not None:
                d[name] = self[name]
        return d



# TODO split AuditoryRecording and Recording for general
class RecordingAuditory(BaseEfysDir):

    _filteredrecordingdirname = 'filtered'
    _stimulidirname = 'stimuli'
    _stimulustablename = 'stimulustable.csv'
    _figuredirname = 'figures'

    def __init__(self, path):
        BaseEfysDir.__init__(self, path=path)
        self._filteredpath = self._path / self._filteredrecordingdirname
        if not self._filteredpath.exists():
            self._filteredpath.mkdir()
        self._filteredsignals = {}
        self._update_filteredsignals()
        self._figurepath = self._path / self._figuredirname
        self._stimulipath = self._path / self._stimulidirname
        self._stimulustablepath = self._stimulipath / self._stimulustablename
        if not self._stimulipath.exists():
            self._stimulipath.mkdir()
        if not self._figurepath.exists():
            self._figurepath.mkdir()
        self._hasstimuli = False
        self._stimuli = None
        if (self._stimulustablepath).exists():
            try:
                self._hasstimuli = True
                self._stimuli = AuditoryStimuli(self._stimulipath)
            except:
                warnings.warn(f'stimulustable exists, but cannot init a AuditoriStimuli object')



    @property
    def figurepath(self):
        return self._figurepath

    @property
    def stimulipath(self):
        return self._stimulipath

    @property
    def stimulustablepath(self):
        return self._stimulustablepath

    @property
    def stimulustable(self):
        if self._stimulustablepath.exists():
            return pd.read_csv(self._stimulustablepath)
        else:
            return None

    @property
    def hasstimuli(self):
        if self._stimuli is not None:
            return True
        else:
            return False

    @property
    def stimuli(self):
        return self._stimuli


    @property
    def filteredpath(self):
        return self._filteredpath

    @property
    def filteredsignals(self):
        return self._filteredsignals

    @property
    def filteredsignalnames(self):
        return sorted(list(self._filteredsignals.keys()))

    # to be implemented by subclass
    @contextmanager
    def open_raw(self):
        # yields a uts MultiChannelUniformTimeSeries in subclass
        yield None

    # to be implemented by subclass
    def load_raw(self):
        return None

    # to be implemented by subclass
    @contextmanager
    def open_bitsnd(self):
        yield None

    # to be implemented by subclass
    def load_bitsnd(self):
        return None

    def _update_filteredsignals(self):
        for sname in sorted(self._filteredpath.glob('*')):
            path = self._filteredpath / sname
            try:
                s = uts.opendarr(path, 'r')
                self._filteredsignals[sname.name] = s
            except:
                warnings.warn(f'{sname} has a directory, but cannot instantiate filtered signal')

    def _update_recordingclass(self, classname):
        if not classname in recordingclasses:
            raise ValueError(f"Recording class `{classname}` does not exist.")
        self._update_infofile(classname=classname)

    def create_lplfp(self, freq=200., phase='zero', newfs=1000.,
                   signalname='lplfp', threads=4, reportprogress=True, overwrite=False):
        """LFP filter based on low-pass filtering only

        Filter characteristics are designed by mne, but filtering is done
        by uts for memory efficiency.

        Parameters
        ----------
        freq
        phase
        newfs
        signalname
        overwrite

        Returns
        -------
        MultiChannelUniformTimeSeries

        """
        outputpath = self.filteredpath / signalname
        if not overwrite and outputpath.exists():
            print(f'{outputpath} exists and overwrite is False, skipping...')
            return
        with self.open_raw() as raw:
            lfp = create_lplfp(s=raw, path=outputpath, freq=freq, phase=phase, newfs=newfs,
                               threads=threads, reportprogress=reportprogress, overwrite=overwrite)
        self._update_filteredsignals()
        return lfp

    def create_bplfp(self, freqs=(0.5, 200.), phase='zero', newfs=1000.,
                   signalname='bplfp', threads=4, reportprogress=True, overwrite=False):
        """LFP filter based on band-pass filtering

        Filter characteristics are designed by mne, but filtering is done
        by uts for memory efficiency.

        Parameters
        ----------
        freqs:
            tuple (low, high)
        phase
        newfs
        signalname
        overwrite

        Returns
        -------
        MultiChannelUniformTimeSeries

        """
        outputpath = self.filteredpath / signalname
        if not overwrite and outputpath.exists():
            print(f'{outputpath} exists and overwrite is False, skipping...')
            return
        with self.open_raw() as raw:
            lfp = create_bplfp(s=raw, path=outputpath, freqs=freqs, phase=phase, newfs=newfs,
                               threads=threads, reportprogress=reportprogress, overwrite=overwrite)
        self._update_filteredsignals()
        return lfp

    def create_filteredsignals(self, threads=4, overwrite=False):
        """Create standard filtered signals: bplfp, lplfp, amua, esa

        """
        s1 = self.create_bplfp(threads=threads, overwrite=overwrite)
        s2 = self.create_lplfp(threads=threads, overwrite=overwrite)
        s3 = self.create_amua(threads=threads, overwrite=overwrite)
        s4 = self.create_esa(threads=threads, overwrite=overwrite)
        return [s1, s2, s3, s4]

    def create_esa(self, signalname='esa', threads=None, reportprogress=True,
                   overwrite=False):
        outputpath = self.filteredpath / signalname
        if not overwrite and outputpath.exists():
            print(f'{outputpath} exists and overwrite is False, skipping...')
            return
        with self.open_raw() as raw:
            esa = create_esa(raw, path=outputpath,
                               threads=threads, reportprogress=reportprogress,
                               overwrite=overwrite)
        self._update_filteredsignals()
        return esa

    def create_amua(self, signalname='amua', threads=4, reportprogress=True, overwrite=False):
        outputpath = self.filteredpath / signalname
        if not overwrite and outputpath.exists():
            print(f'{outputpath} exists and overwrite is False, skipping...')
            return
        with self.open_raw() as raw:
            amua = create_amua(raw, self.filteredpath / signalname,
                               threads=threads, reportprogress=reportprogress,
                               overwrite=overwrite)
        self._update_filteredsignals()
        return amua

    def filteredsend2trash(self, signalname):
        path = self._filteredpath / signalname
        if not path.exists():
            raise IOError(f"signal '{signalname}' does not exist ({self.filteredsignalnames})")
        send2trash.send2trash(str(path))


class RecordingPMM2015(RecordingAuditory):

    _rawrelpath = 'recordingdata.darr'
    _bitsndrelpath = 'sound.darr'

    @property
    def rawpath(self):
        return self.path / self._rawrelpath

    @property
    def bitsndpath(self):
        return self.path / self._bitsndrelpath


    # to be implemented by subclass
    @contextmanager
    def open_raw(self):
        yield uts.opendarr(self.rawpath)
    # to be implemented by subclass
    def load_raw(self):
        return uts.opendarr(self.rawpath)

    # to be implemented by subclass
    @contextmanager
    def open_bitsnd(self):
        yield uts.opendarr(self.bitsndpath)

    # to be implemented by subclass
    def load_bitsnd(self):
        return uts.opendarr(self.bitsndpath)


class RecordingSL2020(RecordingAuditory):

    _LPLFPNAME = 'lplfp.darr'
    _BPLFPNAME = 'bplfp.darr'
    _AMUANAME = 'amua.darr'
    _ESANAME = 'esa.darr'

    def __init__(self, path):

        RecordingAuditory.__init__(self, path=path)

        h5files = [f for f in self._path.glob('*.h5')]
        if not len(h5files) == 1:
            raise ValueError(f"There should be a single h5 file in this folder ({path}),"
                             f"but this is not the case ({h5files})")
        self._rawpath = h5files[0]
        self._amuapath = self._filteredpath / self._AMUANAME
        self._lplfppath = self._filteredpath / self._LPLFPNAME
        self._bplfppath = self._filteredpath / self._BPLFPNAME
        self._esapath = self._filteredpath / self._ESANAME

    @property
    def hasamua(self):
        return self._amuapath.exists()

    @property
    def haslplfp(self):
        return self._lplfppath.exists()

    @property
    def hasbplfp(self):
        return self._bplfppath.exists()

    @property
    def hasesa(self):
        return self._esapath.exists()


    def __str__(self):
        return f"<Recording: {self.path}>"

    __repr__ = __str__

    @contextmanager
    def open_raw(self):
        with mcsh5.open_nn8x8rev5recording(self._rawpath) as (raw, bitsnd):
            yield raw

    def load_raw(self):
        with self.open_raw() as raw:
            raw = raw[:]
        return raw

    @contextmanager
    def open_bitsnd(self):
        with mcsh5.open_nn8x8rev5recording(self._rawpath) as (raw, bitsnd):
            yield bitsnd

    def load_bitsnd(self):
        with self.open_bitsnd() as bitsnd:
            bitsnd = bitsnd[:]
        return bitsnd

    @property
    def lplfp(self):
        if self._lplfppath.exists():
            return uts.opendarr(self._lplfppath)
        else:
            return None

    @property
    def bplfp(self):
        if self._bplfppath.exists():
            return uts.opendarr(self._bplfppath)
        else:
            return None

    @property
    def esa(self):
        if self._esapath.exists():
            return uts.opendarr(self._esapath)
        else:
            return None

    @property
    def amua(self):
        if self._amuapath.exists():
            return uts.opendarr(self._amuapath)
        else:
            return None


    def stimulusspectrogram(self, starttime, endtime, nperseg=512, noverlap=256,
                            dynrange=40, ax=None, ylim=(0, 8000), labels=True,
                            kHz=True, title=None):
        """Creates spectrogram of *reconstructed* stimulation sound

        Reconstructed means that it is based on the playback stimuli (not
        recorded stimuli) and the timing of those stimuli as obtained from
        the recordingdata stimulus table.

        Parameters
        ----------
        starttime
        endtime
        nperseg
        noverlap
        dynrange
        ax
        ylim

        Returns
        -------
        ax, f, t, Sxx
            ax: figure axes
            f: sampling frequencies
            t: sampling times
            Sxx: power as a function of f and t (spectrogram)

        """

        from scipy.signal import spectrogram

        snd, fs = self.stimuli.read_reconstructedsoundrecordingwav()
        f, t, Sxx = spectrogram(snd[int(starttime * fs):int(endtime * fs)], fs=fs,
                                nperseg=nperseg, noverlap=noverlap, window='hanning')
        Sxx[Sxx == 0.0] = np.finfo('float64').eps
        maxdb = 10 * np.log10(Sxx).max()
        if ax is None:
            ax = plt.gca(frameon=False)
        fmin = f[0]
        fmax = f[-1]
        if kHz:
            fmax /= 1000
            fmin /= 1000
            ylabel = 'Frequency (kHz)'
        else:
            ylabel = 'Frequency (Hz)'
        ax.imshow(10 * np.log10(Sxx), origin='lower', cmap='gray_r',
                   extent=(t[0] + starttime, t[-1] + starttime, fmin, fmax), aspect='auto',
                   clim=(maxdb - dynrange, maxdb))
        _ = plt.ylim(ylim)
        _ = plt.ylabel(ylabel)

        _ = plt.xlabel('Time (s)')
        if title is not None:
            _ = plt.title(title)

        if labels:
            st = self.stimulustable
            stimuli = st[(st['starttime'] > starttime) & (st['endtime'] < endtime)]
            color = '#ff5500'
            for i, d in stimuli.iterrows():
                midtime = (d['endtime'] + d['starttime']) / 2.
                label = d['snd']
                plt.text(midtime, ylim[1] - (ylim[1] - ylim[0]) / 100, label, ha='center', va='top',
                         color=color, fontsize='large', fontweight='bold')
                plt.plot((d['starttime'], d['starttime']), ylim, color=color, lw=0.5)
                plt.plot((d['endtime'], d['endtime']), ylim, color=color, lw=0.5)
        return ax, f, t, Sxx


recordingclasses = {
    'RecordingAuditory': RecordingAuditory,
    'RecordingPMM2015' : RecordingPMM2015,
    'RecordingSL2020': RecordingSL2020,
}

