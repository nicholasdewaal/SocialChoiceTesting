from numpy import cumsum, array, intc, random
from random import uniform, randint
from collections import defaultdict
# from ipdb import set_trace
from functools import lru_cache
from pref_matrix.pref_matrix import c_gen_pref_summaries


@lru_cache(maxsize=32)
def get_weights_from_counts(counts):
    return tuple(tuple(count / sum(cnt_row) for count in cnt_row)
                 for cnt_row in counts)


def assert_weights_sound(weights):
    # ranked_weights must be balanced sum to 100% along rows and columns
    for row in weights:
        assert abs(sum(row) - 1) < .0001
    for column in zip(*weights):
        assert abs(sum(column) - 1) < .0001


def tuplize(in_arr):
    '''
    Convert array to all tuples to allow for hashing in caching
    '''
    return tuple(tuple(s for s in t) for t in in_arr)


def fast_gen_pref_summ(pref_ballots):
    '''
    A wrapper around the Cython function that sets up the data to the
    appropriate data structure, and converts return values to tuples.
    '''
    p = array(pref_ballots, dtype=intc)
    n_pref_rk, n_pref_ij = c_gen_pref_summaries(p)
    # must return a tuple to be hashable for caching
    # return tuplize(n_pref_rk), tuplize(n_pref_ij)
    return n_pref_rk, n_pref_ij


def gen_until_2_winners(pref_ballots, method='borda', borda_decay=.5,
                        points_to_win=2.3):
    '''
    method = iterated_borda, borda, iterated_borda_decay, borda_decay, or
    plurality

    borda_decay is the number of points assigned to a winner at the median
    of a voter's ballot.

    Return value: the set of two primary election winners from index
    1..num_candidates.

    This method chooses 2 primary winners using a variation on the random
    ballot where randomly chosen ballots are selected until at least 2
    candidates have sufficiently many points=points_to_win to win from the
    combined randomly sampled ballots.

    The points required to be a primary winner are tallied depending on the
    method.

    If method='iterated_borda':
        The 1st ballot randomly chosen has that voter's 1st choice candidate
        receive 1 point. The ith ballot randomly chosen has that voter's jth
        (j=i%n) choice candidate receive (n - i - 1) / (n - 1) for the i-th
        choice of that ballot.

    If method='iterated_borda_decay':
        The 1st ballot randomly chosen has that voter's 1st choice candidate
        receive 1 point. The 2st ballot randomly chosen has that voter's 2st
        choice candidate receive decay_rate ** 1 points.
        After the number of ballots is sampled reaches the number of
        ranks/candidates, the process is repeated until 2 primary winners
        emerge.

    If method='borda':
        The ballot randomly chosen awards each candidate points according to
        borda count from that ballot with 1 point for the top down to
        (n - i - 1) / (n - 1) for the i-th choice of that ballot and n
        candidates.

    If method='borda_decay':
        receive decay_rate ** (k - 1) points for the kth preference of the
        randomly selected ballot.

    If method='plurality':
        Points are awarded by the first choice pick of each randomly selected
        ballot.

    The first 2 candidates to pass the threshold of points_to_win, win the
    primary to pass on to the final election.
    '''
    won_pts = defaultdict(int)
    win_set = set()
    n_cand = len(pref_ballots[0])
    iter_num = 0
    decay_rate = borda_decay ** (2 / (n_cand - 1))
    rand_max = len(pref_ballots) - 1
    pool_size = int(2 * n_cand ** 1.4 * points_to_win)
    set_length = 0

    if method == 'borda_decay' or method == 'iterated_borda_decay':
        point_legend = [decay_rate ** ii for ii in range(n_cand)]
    elif method != 'plurality':
        # 1st choice gets 1 point scaled down to 0 for the last
        point_legend = [(n_cand - ii - 1) / (n_cand - 1) for ii in range(n_cand)]
    borda = True if method == 'borda' or method == 'borda_decay' else False

    while set_length < 2:
        idx = iter_num % pool_size
        if idx == 0: # much faster than randint
            random_int_pool = random.randint(rand_max, size=pool_size)
        # chosen_ballot = pref_ballots[randint(0, rand_max)]
        chosen_ballot = pref_ballots[random_int_pool[idx]]

        # Iterate between ballots choosing 1st choice on the 1st randomly
        # chosen ballot, the 2nd choice on the 2nd randomly chosen ballot, etc.
        # until cycling back to the 1st choice assigning points according to
        # borda count based on ballot rank.
        if borda:
            for ii, x in enumerate(chosen_ballot):
                won_pts[x] += point_legend[ii]
                if won_pts[x] >= points_to_win:
                    win_set.add(x)
        else:
            if method != 'plurality':
            # then it must be iterated_borda_decay or iterated_borda
                i = iter_num % n_cand
                winner = chosen_ballot[i]
                won_pts[winner] += point_legend[i]
            else: # definitely plurality
                winner = chosen_ballot[0]
                won_pts[winner] += 1

            if won_pts[winner] >= points_to_win:
                win_set.add(winner)


        set_length = len(win_set)
        if set_length > 2:
            win_set = set(sorted(won_pts, key=won_pts.get)[:2])
            # Two winners with the most points that pass points_to_win
        iter_num += 1

    return win_set


def get_pairoff_winner(two_candidates, pref_ij):
    '''
    Out of a set of two_candidates, choose the winner in a pair-off given the
    voter preferences pref_ij of candidate i over candidate j.
    '''
    primary_winners = list(two_candidates)
    if pref_ij[primary_winners[0]][primary_winners[1]] > \
            pref_ij[primary_winners[1]][primary_winners[0]]:
        return primary_winners[0]
    return primary_winners[1]


def random_ballot(pref_ballots):
    '''
    Generate winner by randomly selecting a ballot and choosing their first
    choice
    '''
    idx_winner = randint(0, len(pref_ballots) - 1)
    winner = pref_ballots[idx_winner][0]
    return winner


def multi_lottery(pref_ballots, points_to_win=2.3, borda_decay=.5,
                  pref_ij=None, n_pref_by_rank=None, method='borda'):
    '''
    Returns (set of primary winners (2), finals winner)
    '''
    if (pref_ij is None):
        _, pref_ij = fast_gen_pref_summ(pref_ballots)
    win2_set = gen_until_2_winners(pref_ballots, method=method,
                                   points_to_win=points_to_win)

    return win2_set, get_pairoff_winner(win2_set, pref_ij)
