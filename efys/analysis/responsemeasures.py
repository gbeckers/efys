import random
import numpy as np


def randomlypairedindices(n, remove_same=True):
    """Generate an index of size n, and a ramdomly shuffled version of it

    n: size of index
    remove_same: remove indices that happen to end up in the same position

    """
    ind = np.arange(n)
    ind_r = ind.copy()
    random.shuffle(ind_r)
    if remove_same:
        m = (ind != ind_r)
        return ind[m], ind_r[m]
    else:
        return ind, ind_r


def responsestereotypy(signal, stimulustable, epochduration, preepochduration, nruns=5, weightbycount=True,
                        baseonmax=False):
    """Response stereotypy per channel.

    Calculates how stereotypic responses are to the same stimulus,
    based on the corrlation coeffcicient between each epoch and a
    random other epoch (excluding itself).

    Parameters
    ----------
    signal: MultiChannelUniformTime object
    stimulustable: pandas stimulus table
    epochduration: duration of the epoch to be considered
    nruns: how many times should an epoch be compared to a random other epoch?

    Returns
    -------
    An array of size nchannels, the order or which is determined
    by the signal.channelnames order.

    """
    snds = stimulustable.snd.unique()  # unique sound labels in table
    cc = []  # list of lists with corrcoef (=stereotypy) per channel
    weights = []
    for snd in snds:
        epochs = stimulustable[stimulustable.snd == snd]  # select epochs of same sound
        emcuts = signal.get_epochs(duration=epochduration+preepochduration, epochs=epochs.to_records(),
                                   origintime=preepochduration)
        for runn in range(nruns):
            cc.append([])
            ind, ind_r = randomlypairedindices(n=emcuts.nepochs, remove_same=True)
            weights.append(emcuts.nepochs)  # average is weighted by number of sounds per type
            for chn in emcuts.channelnames:
                cc[-1].append(np.corrcoef(emcuts[:, [chn], ind].samples[:].flatten(),
                                          emcuts[:, [chn], ind_r].samples[:].flatten())[0][1])
    if weightbycount is False:
        weights = None
    if baseonmax:
        return np.max(np.array(cc), axis=0)
    else:
        return np.average(np.array(cc), axis=0, weights=weights)



def z_score(a, b):
    """Calculates z-score of responsiveness between samples of two variables.

    Parameters a and b contain the two variables. If their dimension is larger
    than 1, the z-scores are calculated over the first axis (0).

    """

    ma = np.nanmean(a, 0)
    mb = np.nanmean(b, 0)
    sda = np.nanstd(a, 0)
    sdb = np.nanstd(b, 0)
    covab = np.nanmean((a - ma[np.newaxis, ...]) * (b - mb[np.newaxis, ...]), 0)

    return (ma - mb) / np.sqrt(sda ** 2 + sdb ** 2 - 2 * covab)