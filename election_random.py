
from addict import Dict
from multiprocessing import Pool
from numpy import cumsum, arange
from random import uniform, shuffle
from collections import defaultdict
from scipy.stats import zipf
import matplotlib.pyplot as plt
from ipdb import set_trace
from itertools import combinations
from functools import partial
# from social_choice.profile import Profile, ballot_box, plurality
import svvamp
import irv_variants
from pref_matrix.pref_matrix import c_gen_pref_summaries
import pandas as pd


def gen_weights_zipf(n_weights, zipf_param=1.13):
    '''
    Generate first choice candidate preference frequencies among voters
    assuming that prefence is zipf distributed. Truncate at n_weights total
    candidates/frequencies.
    '''
    rv = zipf(zipf_param)
    out_weights = [rv.pmf(j) for j in range(1, n_weights+1)]
    reweight_factor = sum(out_weights)
    out_weights = [x / reweight_factor for x in out_weights]
    return out_weights


def gen_ranked_preferences_zipf(n_candidates, n_samples, zipf_param=1.1):
    '''
    Generate ranked choice candidate preference frequencies among voters
    assuming that preference rankings are zipf distributed.
    n_samples might need to be about 500 * n_candidates
    '''
    candidates = list(range(n_candidates))
    pref_ballot_samples = list()

    rv = zipf(zipf_param)
    # zipf of index 0 doesn't exist, thus add 1: ii+1
    scaler = sum(rv.pmf(ii+1) for ii in range(n_samples))
    n_prefs = [round(n_candidates*rv.pmf(i+1)/scaler) for i in range(n_samples)]

    # Generate random preference ordering
    for n in n_prefs:
        tmp_candidates = candidates.copy()
        shuffle(tmp_candidates)
        pref_ballot_samples.extend([tuple(tmp_candidates)]*n)

    return pref_ballot_samples


def get_weights_from_counts(counts):
    return [[count/sum(cnt_row) for count in cnt_row] for cnt_row in counts]


def gen_pref_summaries(pref_ballots): # DELETE ONCE CYTHON FUNC TESTED!

    '''
    Warning: this function is slow. Use the cython implementation in pref_matrix
    n_pref_by_rank: # of voters who placed candidate (col idx) at rank (row idx)
    n_pref_i_over_j: # voters preferring candidate of row i to candidate of col j
    '''
    N = len(pref_ballots[0])
    n_pref_i_over_j = N * [N * [0]]
    n_pref_by_rank =  N * [N * [0]]

    for pref_rank in pref_ballots:
        for jj, ranked_val in enumerate(pref_rank):
            n_pref_by_rank[jj][ranked_val] += 1
            for c_less_pref in pref_rank[jj+1:]:
                n_pref_i_over_j[ranked_val][c_less_pref] += 1 #this line is
                # half the cpu work

    # for pref_rank in pref_ballots: #same speed as above!
        # previous = set()
        # for ranked_val in pref_rank:
            # for p in previous:
                # n_pref_i_over_j[p][ranked_val] += 1 #this line is # half the cpu work
            # previous.add(ranked_val)

    return n_pref_by_rank, n_pref_i_over_j


def get_pairwise_winners(n_pref_i_over_j):
    '''
    pair_by_pair_winners: dict of pairs of candidates, and which would win
    among the voters in a pair-off given existing pref_ballots.
    '''
    pair_by_pair_winners = dict()
    candidates = range(len(n_pref_i_over_j))
    for c in combinations(candidates, 2):
        srt_combo = tuple(sorted(list(c)))
        if n_pref_i_over_j[c[0]][c[1]] >= n_pref_i_over_j[c[1]][c[0]]:
            pair_by_pair_winners[srt_combo] = c[0]
        else:
            pair_by_pair_winners[srt_combo] = c[1]

    return pair_by_pair_winners


def scale_utilities(in_utilities):
    scaled_util = [(x-min(x))/(max(x)-min(x)) for x in in_utilities]
    # scaled to get a percent total satisfaction of a population.
    return sum(scaled_util) / len(scaled_util) # average happiness by candidate


def social_util_by_cand(ranked_weights, fraction_happy_decay=.5):
    '''
    Assume a multiplicative fractional utility decay for a voter
    by each drop in their preference ranking
    '''
    assert fraction_happy_decay < 1 and fraction_happy_decay > 0

    # we want the middle preferred candidate to be fraction_happy_decay
    # as desirable as the best choice. Solve for fraction_happy_decay:
    # decay_rate ** ((L-1) / 2) = fraction_happy_decay

    decay_rate = fraction_happy_decay ** (2 / (len(ranked_weights) - 1))
    happiness = defaultdict(int)

    # loop over top to bottom choice
    for ii, weights in enumerate(ranked_weights):
        # loop over candidates
        for jj, candidate_val in enumerate(weights):
            happiness[jj] += candidate_val * decay_rate ** ii
    max_happiness = max(happiness.values())
    # rescale to 1 the best happiness
    for x in happiness:
        happiness[x] = happiness[x] / max_happiness
    return happiness


def gen_until_2_winners_borda(ranked_weights, points_to_win=2.3,
                              borda_decay=.5):
    '''
    borda_decay is the number of points assigned to a winner at the median
    of a voter's ballot.
    ranked_weights: Index=0 corresponds to the weights frequency of the
    1st ranked choices of candidates by voters, index=1 corresponds to the
    weights frequency of the 2nd ranked choices of candidates by voters,
    etc.

    Return value: the set of two primary election winners from index
    1..num_candidates.
    '''
    #!!! TO DO: assert ranked_weights is balanced sum to 100% along rows.!!!
    assert borda_decay < 1 and borda_decay > 0

    decay_rate = borda_decay ** (2 / (len(ranked_weights) - 1))
    ranges = [list(cumsum(weights)[0:-1]) for weights in ranked_weights]
    won_points = defaultdict(int)
    winning_set = set()
    n_current_winners = 0

    # Select 2 primary winners using a hybrid of Borda count, and a variation
    # of the random ballot where randomly chosen ballots are selected, and a
    # primary candidate is chosen one of the two winners if they win
    # sufficiently many points from each randomly sampled ballot.

    # The 1st ballot randomly chosen has that voter's 1st choice candidate
    # receive 1 point.
    # The 2st ballot randomly chosen has that voter's 2st choice candidate
    # receive 1 / 2 point.
    #  ....
    # When k = number of candidates, the kth ballot randomly chosen has that
    # voter's kth choice candidate receive 1 / 2 ** k points.

    # The first 2 candidates to pass the threshold of points_to_win, win the
    # primary to pass on to the final election.

    while n_current_winners < 2:

        # Select winners randomly with probability proportional to their
        # number of votes ranked at level i preference, awarding
        # 1/2 ** i points to that candidate according to Borda count.

        # Candidates are indexed from 0 to the number of candidates-1 =
        # len(ranked_weights)-1

        for i, rng in enumerate(ranges):
            rn = uniform(0,1)
            next_win = sum(1 for x in rng if x < rn)
            won_points[next_win] += decay_rate ** i
            if won_points[next_win] >= points_to_win and next_win not in winning_set:
                n_current_winners += 1
                winning_set.add(next_win)
            if n_current_winners == 2:
                break

    return winning_set


def gen_until_2_winners_plurality(weights, points_to_win=2):
    '''
    Plurality voting special case. weights are first choice only; not ranked.
    '''
    return gen_until_2_winners_borda([weights], points_to_win)


def get_pairoff_winner(two_candidates, pair_by_pair_winner):
    primary_winners = list(two_candidates)
    return pair_by_pair_winner[tuple(sorted(primary_winners))]


def DeWaal_borda(pref_ballots, points_to_win=2.3, borda_decay=.5,
                 pref_ij=None, n_pref_by_rank=None):
    '''
    Returns (set of primary winners (2), finals winner)
    '''
    if not(pref_ij and n_pref_by_rank):
        n_pref_by_rank, pref_ij = gen_pref_summaries(pref_ballots)
    pairwise_wins = get_pairwise_winners(pref_ij)
    weights = get_weights_from_counts(n_pref_by_rank)
    win2_set = gen_until_2_winners_borda(weights, points_to_win, borda_decay)
    return win2_set, get_pairoff_winner(win2_set, pairwise_wins)


def DeWaal_plurality(pref_ballots, points_to_win=2,
                 pref_ij=None, n_pref_by_rank=None):
    '''
    Returns (set of primary winners (2), finals winner)
    '''
    if not(pref_ij and n_pref_by_rank):
        n_pref_by_rank, pref_ij = gen_pref_summaries(pref_ballots)
    pairwise_wins = get_pairwise_winners(pref_ij)
    weights = get_weights_from_counts(n_pref_by_rank)[0]
    win2_set = gen_until_2_winners_plurality(weights, points_to_win)
    return win2_set, get_pairoff_winner(win2_set, pairwise_wins)


def simulate_DeWaal(weights, pair_by_pair_winner, num_sim_per_weight=1000,
                    n_pts_win=2, choice_func=DeWaal_borda):

    if choice_function == DeWaal_borda:
        assert len(weights) == len(weights[0])

    # More simulations per weight are helpful for more candidate weights
    # to provide statistical accuracy.
    num_sim = num_sim_per_weight * len(weights)
    current_sim = 0
    num_primaries_won = defaultdict(int)
    num_finals_won = defaultdict(int)
    happiness_freqs = list()
    # return_tuple = namedtuple()

    while current_sim < num_sim: # Do num_sim simulated elections
        # Handle simulated primary elections
        primary_winners, finals_winner = choice_func(pref_ballots, n_pts_win)
        for winner in primary_winners:
            num_primaries_won[winner] += 1
        # Handle simulated final elections
        num_finals_won[finals_winner] += 1
        current_sim += 1

    # Since 2 candidates win each primary, the frequecy sums to 2, not 1
    freq_primry_won = {key: float(val) / num_sim for key, val in
                       num_primaries_won.items()}
    freq_finals_won = {key: float(val) / num_sim for key, val in
                       num_finals_won.items()}

    finals_social_happinesses = social_util_by_cand(weights)

    # In the following, happiness_freqs is a list of tuples of social happiness
    # occurring in the election simulation at a frequency of the corresponding
    # to the second element of each tuple: (happiness_measure, frequency in sim)

    final_winners = set(freq_finals_won.keys())
    for candidate_key, happy_ms in finals_social_happinesses.items():
        if candidate_key in final_winners:
            happiness_freqs.append((happy_ms, freq_finals_won[candidate_key]))

    # The following is the average social happiness from the election results
    # in the simulation.
    avg_happiness = sum(h[0] * h[1] for h in happiness_freqs)

    return freq_primry_won, freq_finals_won, happiness_freqs, avg_happiness


def plot_sim(weights, pair_by_pair_winner, test_point_cuttoffs=[1,2,3],
             choice_function=DeWaal_borda):

    n_groups = len(weights)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('xkcd:gray')
    ax.set_facecolor((0.38, 0.34, 0.22))
    index = arange(n_groups)
    bar_width = 0.75 / len(test_point_cuttoffs)
    opacity = 0.8
    colors = 'kbgrcmy'
    num_sim_per_weight = 1500

    for j, pts in enumerate(test_point_cuttoffs):

        freq_primry_won, freq_finals_won, happiness_freqs, avg_happiness = \
            simulate_DeWaal(weights, pair_by_pair_winner, num_sim_per_weight,
                               n_pts_win=pts, choice_function=choice_function)
        print("avg_happiness: ", avg_happiness)
        # print("happiness_distr: ", happiness_freqs)
        print("Final_winner: ", freq_finals_won)

        primary_freqs = [freq_primry_won[ii] for ii in freq_primry_won]
        plt.bar(index + j*bar_width, primary_freqs, bar_width, alpha=opacity,
                color=colors[(j+1)%len(colors)], label=str(pts) + ' ballots')

    plt.xlabel('Candidate')
    plt.ylabel('Probability of Winning Primary (2 Candidates will win)')
    plt.title('Scores by person, method=' + choice_function.__name__)
    plt.xticks(index + bar_width, [str(x) for x in range(n_groups)])
    plt.legend()

    plt.tight_layout()
    plt.show()

    # to do!! plot avg happiness as a function of zipf, num_candidates,
    # point_cutoff.
    # plot happiness distributions
    # plot freq_finals_won

    # plt.savefig('election_plot.png')


def simulate_all_elections(pop_object, fast=False, pref_i_to_j=None,
                           n_pref_by_rank=None):
    '''
    fast = True eliminates time consuming SCF's from testing

    borda-irv: Baldwin, Nanson

    Borda: classic
    Condorcet: classic
    Simpson: choose the candidate whose worst pairwise defeat is better than
    that of all other candidates

    Plurality/FPTP

    Implement Later: Dowdall, Symmetric borda, combine Baldwin
    (borda-irv) with Condorcet smith? Copeland
    Raynaud (remove biggest Condorcet loser # iteratively)

    Multi-optimization objectives for choosing a social-choice function
    should be to:
       1. Minimize election manipulability from candidates, voters or
          coalitions.
       2. Maximize social welfare/utility.
       3. Maximize political stability.

    Sometimes social welfare is less important than maintaining a stable
    government. Highly polarized societies may cause an unstable government if
    a Condorcet winner is chosen; those on the losing polar extreme may work
    hard to undermine government, or cause civil war/unrest. Instead, consensus
    candidates improve stability.

    Condorcet methods in polarized societies cause wolves to eat the sheep in
    majority dominance. Borda-style SCF's cause a more consensus candidate to
    win. Iraqi Shia vs. Sunnis, or US politics from 2012-now are great examples.

    '''
    pref_ballots = pop_object.preferences_rk.tolist()
    if not(n_pref_by_rank and pref_i_to_j):
        p = np.array(pop.preferences_rk, dtype=np.intc)
        n_pref_by_rank, pref_i_to_j = c_gen_pref_summaries(p)
    results = dict() # name each election type
    hare_obj = irv_variants.IRV_Variants(pref_ballots, num_i_to_j=pref_i_to_j)
    results['tideman_hare'] = hare_obj.tideman_hare()
    results['smith_hare'] = hare_obj.smith_hare()
    results['woodall_hare'] = hare_obj.woodall_hare()
    results['benham_hare'] = svvamp.ICRV(pop_object).w
    results['hare'] = svvamp.IRV(pop_object).w

    # profile = Profile.ballot_box(pref_ballots)
    results['simpson'] = profile.ranking(profile.simpson) #minimax
    results['schulze'] = svvamp.Schulze(pop_object).w
    results['borda'] =  svvamp.Borda(pop_object).w
    results['nanson'] =  svvamp.Nanson(pop_object).w # borda-irv below avg
    if not fast:
        results['baldwin'] = svvamp.Baldwin(pop_object).w # borda-irv
        results['range'] = svvamp.RangeVotingAverage(pop_object).w
    results['irv-duels'] = svvamp.IRVDuels(pop_object).w
    results['approval'] = svvamp.Approval(pop_object).w
    results['plurality'] = svvamp.Plurality(pop_object).w
    results['coombs'] = svvamp.Coombs(pop_object).w # Like IRV, but elim most low rank
    # DeWaals method simulated 1 times, in sim, average will show in end.

    _, results['dewaal_borda2'] = DeWaal_borda(pref_ballots, points_to_win=2,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank)
    _, results['dewaal_borda2.3'] = DeWaal_borda(pref_ballots,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank)
    _, results['dewaal_borda3'] = DeWaal_borda(pref_ballots, points_to_win=3,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank)
    _, results['dewaal_borda3.8'] = DeWaal_borda(pref_ballots,
        points_to_win=3.8, pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank)
    _, results['dewaal_borda5'] = DeWaal_borda(pref_ballots, points_to_win=5,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank)
    # _, results['dewaal_plurality'] = DeWaal_plurality(pref_ballots,
        # pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank)

    return results


def get_happinesses_by_method(pop_iterator, fast=False):#=iter_rand_pop_polar):

    num_cpu = multiprocessing.cpu_count()
    lock = multiprocessing.lock()
    num_sim, current_sim = 1500, 0
    utils_by_scf = Dict()
    dataframe_dict = Dict()
    test_num_candidates = [3, 4, 6, 9, 13, 18, 24]

    # modify each sim to run in parallel
    while current_sim < num_sim:
        for n_candidates in test_num_candidates: # simulate for various numbers of candidates
            n_voters = n_candidates * 750
            with Pool(num_cpu) as p: # parallelize, put -1 to save a cpu?
                nxt_sim = partial(next_sim_iter, lock=lock,
                    utils_by_scf=utils_by_scf, n_candidates=n_candidates,
                    current_sim=current_sim)
                p.map(nxt_sim, pop_iterator(n_voters, n_candidates))
            # for pop in pop_iterator(n_voters, n_candidates):
                # p = np.array(pop.preferences_rk, dtype=np.intc)
                # n_pref_by_rank, pref_i_to_j = c_gen_pref_summaries(p)
                # weights = get_weights_from_counts(n_pref_by_rank)
                # utils = social_util_by_cand(weights)
                # winners_by_scf, param = simulate_all_elections(pop, fast=fast,
                    # n_pref_by_rank=n_pref_by_rank, pref_i_to_j=pref_i_to_j)
        current_sim += 1

    # now make dict of DataFrames by paramaters, n_candidates
    for param, v_upper in utils_by_scf:
        for k, v_lower in v_upper:
            dataframe_dict[param][k] = pd.DataFrame.from_dict(v)
            dataframe_dict[param][k].boxplot() # labels? by axis?
            plt.savefig("plot_p=" + str(param) + "_n_cand=" + str(k) + ".png")
            # plot means by n_candidates, param


def next_sim_iter(pop, param, lock, utils_by_scf, n_candidates, current_sim):
    p = np.array(pop.preferences_rk, dtype=np.intc)
    n_pref_by_rank, pref_i_to_j = c_gen_pref_summaries(p)
    weights = get_weights_from_counts(n_pref_by_rank)
    utils = social_util_by_cand(weights)
    winners_by_scf = simulate_all_elections(pop, fast=fast,
        n_pref_by_rank=n_pref_by_rank, pref_i_to_j=pref_i_to_j)
    with lock:
        utils_by_scf[param][n_candidates][current_sim] = \
            {k : utils[v] for k,v in winners_by_scf.items()}


def iter_rand_pop_polar(n_voters, n_candidates, num_polarizations=5):

    polarization_list = [3 * ii for ii in range(num_polarizations)]
    for polarization in polarization_list:
        # using the Von-Mises Fisher model, which represents a polarized culture:
        pop = svvamp.PopulationVMFHypersphere(V=n_voters, C=n_candidates,
              vmf_concentration=polarization)
        yield pop, polarization


def iter_rand_pop_other(n_voters, n_candidates, num_polarizations=5):
    # do other data gen methods
    pop = svvamp.PopulationSpheroid(V=num_voters, C=n_candidates)
    yield pop


def iter_rand_pop_ladder(n_voters, n_candidates, num_polarizations=5,
    ladder_rng=10):

    for n in range(ladder_rng): # Neutral cultures
        pop = svvamp.PopulationLadder(V=n_voters, C=n_candidates, nrungs=n)
        yield pop, ladder_rng


def iter_rand_pop_gauss(n_voters, n_candidates, num_polarizations=5):
    #politcal spectrum:
    pop = svvamp.PopulationGaussianWell(V=num_voters, C=n_candidates,
        sigma=[1], shift=[0])
    yield pop, gauss_vals


def iter_rand_pop_zipf(n_voters, n_candidates,
        zipf_params=arange(1.05,3.05, .12)):
    for zipf_param in zipf_params: # Do for zipf sampled ballots
        ballots = gen_ranked_preferences_zipf(n_candidates, n_voters, zipf_param)
        pop = svvamp.Population(preferences_rk=ballots)
        yield pop, zipf_param


if __name__ == "__main__":

    # Simulate DeWaal_borda and plot
    votes = gen_ranked_preferences_zipf(n_candidates=10, n_voters=10000,
                                        zipf_param=1.2)

    n_preferred_by_rank, n_pref_matrix = c_gen_pref_summaries(votes)
    pairwise_wins = get_pairwise_winners(pref_ij)
    w = get_weights_from_counts(n_preferred_by_rank)
    all_happinesses = social_util_by_cand(w)
    plot_sim(w, pairwise_wins, test_point_cuttoffs=[.9,1,1.5,2,2.1,2.5,3,3.1,3.9],
        choice_function=gen_until_2_winners_borda)


    # Simulate all elections once
    pop = svvamp.PopulationVMFHypersphere(V=15000, C=15, vmf_concentration=4)
    simulate_all_elections(pop)
