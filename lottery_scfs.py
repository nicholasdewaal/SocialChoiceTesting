from numpy import cumsum, array, intc
from random import uniform, randint
from collections import defaultdict
from ipdb import set_trace
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
    Convert numpy 2-D array to all tuples to allow for hashing in caching
    '''
    return tuple(tuple(x) for x in in_arr.tolist())


def fast_gen_pref_summ(pref_ballots):
    '''
    A wrapper around the Cython function that sets up the data to the
    appropriate data structure, and converts return values to tuples.
    '''
    p = array(pref_ballots, dtype=intc)
    n_pref_rk, n_pref_ij = c_gen_pref_summaries(p)
    # must return a tuple to be hashable for caching
    return tuplize(n_pref_rk), tuplize(n_pref_ij)


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

    while len(win_set) < 2:
        chosen_ballot = pref_ballots[randint(0, len(pref_ballots) - 1)]
        i = iter_num % n_cand
        iter_num += 1

        if method == 'plurality':
            won_pts[chosen_ballot[0]] += 1

        # Iterate between ballots choosing 1st choice on the 1st randomly
        # chosen ballot, the 2nd choice on the 2nd randomly chosen ballot, etc.
        # until cycling back to the 1st choice assigning points according to
        # borda count based on ballot rank.
        elif method == 'iterated_borda':
            won_pts[chosen_ballot[i]] += (n_cand - i - 1) / (n_cand - 1)

        elif method == 'iterated_borda_decay':
            won_pts[chosen_ballot[i]] += decay_rate ** i

        for ii, x in enumerate(chosen_ballot):
            if method == 'borda':
                # 1st choice gets 1 point scaled down to 0 for the last
                won_pts[x] += (n_cand - ii - 1) / (n_cand - 1)

            if method == 'borda_decay':
                won_pts[x] += decay_rate ** ii

            if won_pts[x] >= points_to_win:
                win_set.add(x)

        win_set = {x for x, v in won_pts.items() if v >= points_to_win}
        if len(win_set) > 2:
            win_set = set(sorted(won_pts, key=won_pts.get)[:2])
            # Two winners with the most points that pass points_to_win

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
    # Attempt 2
    if (pref_ij is None):
        _, pref_ij = fast_gen_pref_summ(pref_ballots)
    win2_set = gen_until_2_winners(pref_ballots, method=method,
                                   points_to_win=points_to_win)

    return win2_set, get_pairoff_winner(win2_set, pref_ij)
