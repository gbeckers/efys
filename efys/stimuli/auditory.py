import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample
import soundfile as sf
import uts, utsviz
import matplotlib.pyplot as plt
from darr import DataDir
from efys.utils import datetimestring
from efys.audiocalibmarks import create_recordingeventtable


class AuditoryStimuli:
    _playbackdirname = 'playback'
    _reconstructedsoundrecordingname = 'reconstructedsoundrecording.wav'
    _stimulustablename = 'stimulustable.csv'

    def __init__(self, path):
        self._path = Path(path)
        self._stimulustablepath = self.path / self._stimulustablename
        playbackpath = (self._path / self._playbackdirname)
        if not playbackpath.exists():
            playbackpath.mkdir()
        self._playbackpath = playbackpath
        self._playbackwavpath, self._playbackcsvpath = self._find_playbackfiles()
        self._reconstructedsoundrecordingwavpath = self._path / self._reconstructedsoundrecordingname
        self._datadir = DataDir(self._path)

    @property
    def path(self):
        return self._path

    @property
    def datadir(self):
        return self._datadir

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
    def playbackwavpath(self):
        return self._playbackwavpath

    @property
    def playbackcsvpath(self):
        return self._playbackcsvpath

    @property
    def playbackpath(self):
        return self._playbackpath

    def _find_playbackfiles(self):
        files = [path.name for path in self._playbackpath.glob('*')
                 if path.suffix in ('.csv', '.wav') and not path.is_dir()]
        wavfiles = [fn for fn in files if Path(fn).suffix == '.wav']
        if len(wavfiles) != 1:
            raise ValueError(f"{self._path} should contain one '.wav' file only")
        wavfilename = wavfiles[0]
        csvfilepath = self._playbackpath / Path(wavfilename).with_suffix('.csv')
        if not csvfilepath.exists():
            raise ValueError(f"'{csvfilepath}' does not exist.")
        return self._playbackpath / wavfilename, csvfilepath

    @property
    def playbackstimulustable(self):
        return pd.read_csv(self._playbackcsvpath)

    def read_playbackwav(self):
        return sf.read(self._playbackwavpath)

    def read_reconstructedsoundrecordingwav(self):
        return sf.read(self._reconstructedsoundrecordingwavpath)

    def create_recordingstimulustable(self, recording, searchduration=35.,
                                      correctones=None, checkcalibmarks=True,
                                      copycheckcalibmarkspath=None, saveparams=True):
        recordedsnd = recording.load_bitsnd()
        playbackstimulustable = self.playbackstimulustable
        frames, fs = self.read_playbackwav()
        playbacksnd = frames
        playbacksnd_fs = float(fs)
        st, params, (fig1, fig2) = create_recordingeventtable(recsnd=recordedsnd.samples[:],
                                                              recsnd_fs=recordedsnd.fs,
                                                              audiostimulustable=playbackstimulustable,
                                                              calibmark=playbacksnd,
                                                              calibmark_fs=playbacksnd_fs,
                                                              recordedasbit=True,
                                                              searchduration=searchduration, bitthreshold=0.005,
                                                              checkcalibmarks=checkcalibmarks,
                                                              correct_ones=correctones)
        if self._stimulustablepath.exists():
            newname = f'{self._stimulustablepath.with_suffix("").name}_old_{datetimestring()}.csv'
            self._stimulustablepath.rename(self._stimulustablepath.with_name(newname))
        st.to_csv(self._stimulustablepath)
        dd = self._datadir
        dd.write_jsondict('timealignmentparameters.json', params, overwrite=True)
        fig1.savefig(dd.path / 'calibmarks.png', dpi=300)
        fig2.savefig(dd.path / 'sndepochs.png', dpi=300)
        if copycheckcalibmarkspath is not None:
            p = str(Path(copycheckcalibmarkspath))
            fig1.savefig(Path(f'{p}_calibmarks.png'), dpi=300)
            fig2.savefig(Path(f'{p}_sndepochs.png'), dpi=300)
        return st

    def reconstruct_soundrecording(self, fs=44100, overwrite=False):
        fs = int(round(fs))
        stt = self.stimulustable
        pbt = self.playbackstimulustable
        pbframes, fs = self.read_playbackwav()
        nframes = int(round(stt.iloc[-1]['endtime'] * fs)) + 2  # +1 for rounding problems, removed later in necessary
        frames = np.zeros(nframes, dtype=pbframes.dtype)
        for (ststartt, stendt, pbstartf, pbendf) in zip(stt.starttime.values, stt.endtime.values,
                                                        pbt.startframe.values, pbt.endframe.values):
            num = int(round((stendt - ststartt) * fs))
            sndframes = resample(pbframes[pbstartf:pbendf], num=num)

            start = int(round(ststartt * fs))
            frames[start:start + num] = sndframes
        if (start + num) != len(frames):
            frames = frames[:-(len(frames) - (start + num))]  # remove trailing zeros
        sf.write(self.path / 'reconstructedsoundrecording.wav', frames, fs)


class StimuliPMM2015(AuditoryStimuli):
    def __init(self, path):
        AuditoryStimuli.__init__(self, path=path)


class StimuliSL2020(AuditoryStimuli):

    def __init(self, path):
        AuditoryStimuli.__init__(self,path=path)
