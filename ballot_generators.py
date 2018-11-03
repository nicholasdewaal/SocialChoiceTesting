from numpy import arange
from random import shuffle
from scipy.stats import zipf
import svvamp


def gen_weights_zipf(n_weights, zipf_param=1.13):
    '''
    Generate first choice candidate preference frequencies among voters
    assuming that prefence is zipf distributed. Truncate at n_weights total
    candidates/frequencies.
    '''
    rv = zipf(zipf_param)
    out_weights = [rv.pmf(j) for j in range(1, n_weights + 1)]
    reweight_factor = sum(out_weights)
    out_weights = [x / reweight_factor for x in out_weights]
    return out_weights


def gen_ranked_preferences_zipf(n_candidates, n_voters, zipf_param=1.1):
    '''
    Generate ranked choice candidate preference frequencies among voters
    assuming that preference rankings are zipf distributed.
    n_voters might need to be about 500 * n_candidates
    '''
    candidates = list(range(n_candidates))
    pref_ballot_samples = list()

    rv = zipf(zipf_param)
    # zipf of index 0 doesn't exist, thus add 1: ii+1
    scaler = sum(rv.pmf(ii + 1) for ii in range(n_voters))
    n_prefs = [n_voters * rv.pmf(i + 1) / scaler for i in range(n_voters)]

    # Generate random preference ordering according to zipf distributed samples
    offset = 0
    for n in n_prefs:
        m = int(round(n + offset))
        offset = n - m + offset
        tmp_candidates = candidates.copy()
        shuffle(tmp_candidates)
        pref_ballot_samples.extend([tuple(tmp_candidates)] * m)

    return tuple(pref_ballot_samples)


def iter_rand_pop_polar(n_voters, n_candidates, num_polarizations=5):
    '''
    This is an iterator that creates num_polarizations populations of voters
    each with a polarization from 0 to 3 * num_polarizations in incremements of
    3 using PopulationVMFHypersphere from the svvamp library.
    '''
    polarization_list = [3 * ii for ii in range(num_polarizations)]
    for polarization in polarization_list:
        # The Von-Mises Fisher model, which represents a polarized culture:
        pop = svvamp.PopulationVMFHypersphere(V=n_voters, C=n_candidates,
                                              vmf_concentration=polarization)
        yield pop, polarization


def iter_rand_pop_other(n_voters, n_candidates, num_param=5):
    '''
    This is an iterator that creates num_param populations of voters
    using PopulationSpheroid from the svvamp library.
    '''
    for i in range(num_param):
        pop = svvamp.PopulationSpheroid(V=n_voters, C=n_candidates)
        yield pop, i


def iter_rand_pop_ladder(n_voters, n_candidates, ladder_rng=10):
    '''
    This is an iterator that creates ladder_rng populations of voters
    using PopulationLadder from the svvamp library with n_rungs from 2 to
    ladder_rng + 1.
    '''
    for n in range(1, ladder_rng + 1):  # Neutral cultures
        pop = svvamp.PopulationLadder(V=n_voters, C=n_candidates, n_rungs=n)
        yield pop, n


def iter_rand_pop_gauss(n_voters, n_candidates, num_param=5):
    '''
    This is an iterator that creates 3 * num_param populations of voters
    using PopulationGaussianWell from the svvamp library with a shift of
    -.5, 0 and .5, with sigma from 1 to num_param. This should later be
    modified to include far more combinations of ranges/shifts together.
    '''
    # politcal spectrum:
    for shft in [-.5, 0, .5]:  # later include more sigmas/shifts combined
        for i in range(1, num_param + 1):
            pop = svvamp.PopulationGaussianWell(V=n_voters, C=n_candidates,
                                                sigma=[i], shift=[shft])
            gauss_vals = (shft, i)
            yield pop, gauss_vals


def iter_rand_pop_zipf(n_voters, n_candidates,
                       zipf_params=arange(1.05, 3.25, .2)):
    '''
    This is an iterator that creates populations of voters using
    gen_ranked_preferences_zipf with the distribution of rank ordering of
    ballots following a truncated zipf distribution with each parameter from
    zipf_params associated with each iterated population of voters.
    '''
    for zipf_param in zipf_params:  # Do for zipf sampled ballots
        ballots = gen_ranked_preferences_zipf(n_candidates, n_voters,
                                              zipf_param)
        pop = svvamp.Population(preferences_rk=ballots)
        yield pop, zipf_param
