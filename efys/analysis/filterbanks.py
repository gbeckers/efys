import numpy as np
import scipy.signal

__all__ = ['gaussianband', 'gaussianfilterbank','gaussianfilterbanklinear']

# FIXME write tests
def gaussianband(centerfreq, bandwidth, freqs, printinfo=False):
    """Create a gausian band for frequency domain multiplication.

    Assumes an rfft.

    Parameters
    ----------
    centerfreq : float
      Peak frequency (i.e. position of max of gaussian) in Hz.)
    bandwidth : float
      Bandwidth (= SD) of gaussian in Hz.
    freqs : Array
      Sequence of frequency bins in Hz. Should be uniformly sampled.
      Typically this is the output of numpy.fft.rfftfreq, which calculates
      the frequency bins given the length of a time signal.
    printinfo : bool
      If True will print some calculated parameters. Nice for debugging

    Returns
    -------
    freqs, window: (Array, Array)
      The gaussian window that can be used for multiplication with the rfft
      of the signal, as well as the frequency values of its bins.

    """
    n_freqbins = len(freqs)
    if printinfo:
        print(f'n_freqbins: {n_freqbins}')
    freqspacing = np.diff(freqs).mean()
    if printinfo:
        print(f'freqspacing: {freqspacing}')
    n_std = round(bandwidth / freqspacing)  # bandwidth gaussion in points
    if printinfo:
        print(f'std of gaussian window in points: {n_std}')
    centerfreqbin = int(round(centerfreq / freqspacing))
    if printinfo:
        print(f'centerfreqbin: {centerfreqbin}')
    if printinfo:
        print(f'requested centerfreq: {centerfreq}, '
              f'actual centerfreq: {freqs[centerfreqbin]}')
    halfM = n_freqbins - centerfreqbin
    if printinfo:
        print(f'halfM: {halfM}')
    window = scipy.signal.gaussian(M=2 * halfM, std=n_std, sym=False)  # full window
    return window[halfM - centerfreqbin:][:n_freqbins].copy()  # cropped to existing freqs

def gaussianfilterbanklinear(startbandfreq, endbandfreq, nbands, bandwidth, freqs,
                       printinfo=False):
    filterbank = np.zeros((nbands, len(freqs)), dtype='float64')
    if printinfo:
        print(f'created empty filterbank: {filterbank.shape}')
    for bandno, cf in enumerate(np.linspace(startbandfreq, endbandfreq, nbands)):
        filterbank[bandno,:] = gaussianband(centerfreq=cf,
                                            bandwidth=bandwidth,
                                            freqs=freqs,
                                            printinfo=printinfo)
        if printinfo:
            print(f'created band: {bandno}')

    return filterbank

def gaussianfilterbanklinear(startbandfreq, endbandfreq, nbands, bandwidth, freqs,
                       printinfo=False):
    filterbank = np.zeros((nbands, len(freqs)), dtype='float64')
    if printinfo:
        print(f'created empty filterbank: {filterbank.shape}')
    for bandno, cf in enumerate(np.linspace(startbandfreq, endbandfreq, nbands)):
        filterbank[bandno,:] = gaussianband(centerfreq=cf,
                                            bandwidth=bandwidth,
                                            freqs=freqs,
                                            printinfo=printinfo)
        if printinfo:
            print(f'created band: {bandno}')

    return filterbank

def gaussianfilterbank(centerfreqs, bandwidths, freqs, printinfo=False):

    if not len(centerfreqs) == len(bandwidths):
        raise ValueError(f"number of centerfreqs and bandwidths needs to be equal "
                         f"(now {len(centerfreqs)} vs {len(bandwidths)})")
    filterbank = np.zeros((len(centerfreqs), len(freqs)), dtype='float64')
    if printinfo:
        print(f'created empty filterbank: {filterbank.shape}')
    for bandno, (cf, bw) in enumerate(zip(centerfreqs, bandwidths)):
        filterbank[bandno,:] = gaussianband(centerfreq=cf,
                                            bandwidth=bw,
                                            freqs=freqs,
                                            printinfo=printinfo)
        if printinfo:
            print(f'created band: {bandno}')
    return filterbank


