# -*- coding: utf-8 -*-
import numpy as np

from typing import List, Tuple, Optional, Iterable
from scipy.spatial.distance import pdist, cdist
from scipy.stats import kstwobign, pearsonr
from scipy.stats import genextreme

__all__ = ['ks2d2s', 'estat', 'estat2d']


def ks2d2s(x1: np.ndarray,
           y1: np.ndarray,
           x2: np.ndarray,
           y2: np.ndarray,
           nboot: Optional[int] = None,
           extra: Optional[bool] = False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples. 
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    nboot: int or None, Optional
        The number of bootstrapping iterations to be performed
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.

    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.

    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the
    p-value is only an approximation as the analytic distribution is unknown. The approximation is accurate enough when
    N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that
    the two samples are not significantly different. (cf. Press 2007)

    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical
    Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of
    the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''

    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = np.random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = np.random.choice(n, n1, replace=True)
            #ix2 = np.random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    """
    Find the average maximum distance between the two maximal distances calculated using maxdist

    Parameters
    ----------
    x1: 1D numpy array of floats
        The x co-ordinate of all the points for set 1
    y1: 1D numpy array of floats
        The y co-ordinate of all the points for set 1
    x2: 1D numpy array of floats
        The x co-ordinate of all the points for set 2
    y2: 1D numpy array of floats
        The y co-ordinate of all the points for set 2

    Returns
    -------
    distance: float
        The average maximal distance

    """
    d1 = maxdist(x1, y1, x2, y2)
    d2 = maxdist(x2, y2, x1, y1)
    return (d1 + d2) / 2


def maxdist(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    """

    Parameters
    ----------
    x1: 1D numpy array of floats
        The x co-ordinate of all the points for set 1
    y1: 1D numpy array of floats
        The y co-ordinate of all the points for set 1
    x2: 1D numpy array of floats
        The x co-ordinate of all the points for set 2
    y2: 1D numpy array of floats
        The y co-ordinate of all the points for set 2

    Returns
    -------
    maximum_distance: float

    """
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x: float, y: float, xx: np.ndarray, yy: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Find the proportion of the sample points in each quadrant compared to the chosen point

    Parameters
    ----------
    x: float
        The x co-ordinate of the chosen point
    y: float
        The y co-ordinate of the chosen point
    xx: 1D numpy array of floats
        The x co-ordinate of all the points
    yy: 1D numpy array of floats
        The y co-ordinate of all the points

    Returns
    -------
    quadrant_values: tuple of floats
        The proportions of the sample points in each quadrant compared to the chosen point

    """
    n = xx.shape[0]
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x: Iterable[float],
          y: Iterable[float],
          nboot: Optional[int] = 1000,
          replace: Optional[bool] = False,
          method: Optional[str] = 'log',
          fitting: Optional[bool] = False) -> Tuple:
    '''
    Energy distance statistics test.

    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist
    '''

    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: np.random.randint(x, size=x)
    else:
        rand = np.random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x: List[float], y: List[float], method: Optional[str] = 'log') -> float:
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n**2 - dy.sum() / m**2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z
