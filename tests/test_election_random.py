
from pref_matrix.pref_matrix import c_gen_pref_summaries
import irv_variants as irv
import simulated_elections as se
import ballot_generators as bg
import lottery_scfs as ls
import pytest
from numpy import array, intc
from svvamp import PopulationSpheroid
# to do: fix imports, create packages, __init__ files


def gen_pref_summaries(pref_ballots):
    '''
    This function is used for testing code to always check it matches the
    cython code version of use. Do not use except for testing.
    This function is slow. Use the Cython implementation in pref_matrix
    n_pref_by_rank: # voters who placed candidate (col idx) at rank (row idx)
    n_pref_i_over_j: # voters preferring candidate row i to candidate col j
    '''
    N = len(pref_ballots[0])
    n_pref_i_over_j = [N * [0] for _ in range(N)]
    n_pref_by_rank = [N * [0] for _ in range(N)]

    for pref_rank in pref_ballots:
        for jj, ranked_val in enumerate(pref_rank):
            n_pref_by_rank[jj][ranked_val] += 1
            for c_less_pref in pref_rank[jj + 1:]:
                n_pref_i_over_j[ranked_val][c_less_pref] += 1  # this line is
                # half the cpu work

    return n_pref_by_rank, n_pref_i_over_j


def test_get_weights_from_counts():
    pref_ballots = [[1, 4, 3, 0, 2],
                    [4, 1, 0, 2, 3],
                    [4, 3, 2, 1, 0],
                    [4, 2, 0, 1, 3],
                    [2, 4, 1, 0, 3],
                    [2, 3, 4, 0, 1],
                    [0, 2, 1, 4, 3],
                    [3, 0, 1, 4, 2],
                    [3, 2, 4, 1, 0],
                    [2, 0, 4, 1, 3]]
    counts, _ = ls.fast_gen_pref_summ(pref_ballots)
    w = ls.get_weights_from_counts(counts)
    wt = list(zip(*w))

    sums = [sum(b) for b in w]
    sumst = [sum(b) for b in wt]
    for s in sums + sumst:
        assert abs(1-s) < .0001
    expected = [(0.1, 0.2, 0.2, 0.3, 0.2),
                (0.1, 0.1, 0.3, 0.4, 0.1),
                (0.3, 0.3, 0.1, 0.1, 0.2),
                (0.2, 0.2, 0.1, 0.0, 0.5),
                (0.3, 0.2, 0.3, 0.2, 0.0)]

    assert wt == expected


def test_gen_pref_summaries():
    for n in range(2, 7):
        pop = PopulationSpheroid(V=100*n, C=n)
        pref_ballots = pop.preferences_rk.tolist()
        p = array(pref_ballots, dtype=intc)
        r1, r2 = c_gen_pref_summaries(p)
        t1, t2 = gen_pref_summaries(pref_ballots)
        assert (t1 == r1).all()
        assert (t2 == r2).all()


def test_gen_weights_zipf():

    for n in [5, 10, 100, 1000]:
        x = bg.gen_weights_zipf(n)
        assert len(x) == n
        assert abs(sum(x) - 1) < .0001
    # add test for distribution?


def test_gen_ranked_preferences_zipf():

    n_candidates = 5
    ordered_ballot = list(range(n_candidates))
    for n_voters in [5, 15, 200, 1000]:
        ballots = bg.gen_ranked_preferences_zipf(n_candidates, n_voters)
        assert len(ballots) == n_voters
        assert len(ballots[0]) == n_candidates
        for b in ballots:
            assert sorted(b) == ordered_ballot
    # add test for distribution?


def test_social_util_by_cand():
    ranked_weights = [[0.1, 0.1, 0.3, 0.2, 0.3],
                      [0.2, 0.1, 0.3, 0.2, 0.2],
                      [0.2, 0.3, 0.1, 0.1, 0.3],
                      [0.3, 0.4, 0.1, 0.0, 0.2],
                      [0.2, 0.1, 0.2, 0.5, 0.0]]
    u = se.social_util_by_cand(ranked_weights)
    assert u[4] == 1
    assert u[0] > u[1]
    ranked_weights = [[0.1, 0.9],
                      [0.9, 0.1]]
    u = se.social_util_by_cand(ranked_weights)
    assert u[0] < u[1]
    assert u[1] == 1

    ranked_weights = [[1.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.0, 0.5, 0.5]]
    u = se.social_util_by_cand(ranked_weights, .5)
    assert u == {0: 1.0, 1: 0.375, 2: 0.375}
    u = se.social_util_by_cand(ranked_weights, .999999999)
    for v in u.values():
        assert abs(v - 1) < .001
    u = se.social_util_by_cand(ranked_weights, 0.00000001)
    assert u[0] == 1
    assert u[1] < .0001
    assert u[2] < .0001

    ranked_weights = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
    for v in [0.1, 0.2, .3, .4, .5, .6, .7, .8, .9]:
        u = se.social_util_by_cand(ranked_weights, v)
        assert u[1] == v
        assert u[0] == 1


def test_get_pairoff_winner():
    # not a legitimate pref matrix, but good enough for tests
    pref_ij = [[1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5],
               [1, 2, 3, 4, 5]]
    for i in range(4):
        assert ls.get_pairoff_winner({i, i+1}, pref_ij) == i

    assert ls.get_pairoff_winner({0, 4}, pref_ij) == 0
    # not a legitimate pref matrix, but good enough for tests
    pref_ij = [[5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1]]
    for i in range(4):
        assert ls.get_pairoff_winner({i, i+1}, pref_ij) == i+1

    assert ls.get_pairoff_winner({0, 3}, pref_ij) == 3


def test_random_ballot():
    n_cand = 5
    n_voters = 10000
    rand_cnts = [0] * n_cand
    counts = [0] * n_cand
    pop = PopulationSpheroid(V=n_voters, C=n_cand)
    p = pop.preferences_rk

    for x in p:
        counts[x[0]] += 1
    pcts = [x / sum(counts) for x in counts]

    for _ in range(1000000):
        x = ls.random_ballot(p)
        rand_cnts[x] += 1
    r_pcts =  [x / sum(rand_cnts) for x in rand_cnts]

    diff = sum([abs(r_pcts[ii] - pcts[ii]) for ii in range(len(r_pcts))])
    assert diff < .005


def test_gen_until_2_winners():
    result = set((0, 1))
    ballots = [[0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [1,0,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4],
               [0,1,2,3,4]]
    assert ls.gen_until_2_winners(ballots, 'borda') == result
    assert ls.gen_until_2_winners(ballots, 'iterated_borda') == result
    assert ls.gen_until_2_winners(ballots, 'iterated_borda_decay') == result
    assert ls.gen_until_2_winners(ballots, 'borda_decay') == result
    assert ls.gen_until_2_winners(ballots, 'plurality') == result


def test_gen_until_2_winners_borda():
    # test that on numerous samples, not all the same result
    pass


def test_gen_until_2_winners_plurality():
    # test that on numerous samples, not all the same result
    pass


def test_multi_lottery_borda():
    # test that on numerous samples, not all the same result
    pass


def test_multi_lottery_plurality():
    # test that on numerous samples, not all the same result
    pass


def standard_iter_tester(pop_iter):
    n_voters = 20
    n_candidates = 4
    param = 5
    if pop_iter == bg.iter_rand_pop_zipf:
        it = pop_iter(n_voters, n_candidates)
    else:
        it = pop_iter(n_voters, n_candidates, param)
    num_items = 0
    for x in it:
        num_items += 1
        prefs = x[0].preferences_rk.tolist()
        assert len(prefs) == n_voters
        assert len(prefs[0]) == n_candidates
        assert x[0].C == n_candidates
        assert x[0].V == n_voters

    if pop_iter == bg.iter_rand_pop_zipf:
        pass

    elif pop_iter == bg.iter_rand_pop_gauss:
        assert num_items == param * 3
    else:
        assert num_items == param

def test_iter_rand_pop_polar():
    standard_iter_tester(bg.iter_rand_pop_polar)


def test_iter_rand_pop_other():
    standard_iter_tester(bg.iter_rand_pop_other)


def test_iter_rand_pop_zipf():
    standard_iter_tester(bg.iter_rand_pop_zipf)


def test_iter_rand_pop_gauss():
    standard_iter_tester(bg.iter_rand_pop_gauss)


def test_iter_rand_pop_ladder():
    standard_iter_tester(bg.iter_rand_pop_ladder)


# Now test all irv

def test_irv():
    p = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    hare_obj = irv.IRV_Variants(p)
    assert hare_obj._primary_smith_set == {0, 1, 2}
    assert hare_obj.hare() in {0, 1, 2}
    assert hare_obj.benham_hare() in {0, 1, 2}
    assert hare_obj.smith_hare() in {0, 1, 2}
    assert hare_obj.tideman_hare() in {0, 1, 2}
    assert hare_obj.woodall_hare() in {0, 1, 2}

    p = [*10 * [[0, 1, 2, 3]],
         [0, 2, 1, 3],
         *9 * [[1, 2, 0, 3]],
         [1, 0, 2, 3],
         *8 * [[2, 0, 1, 3]],
         [2, 1, 0, 3]]
    hare_obj = irv.IRV_Variants(p)
    assert hare_obj._primary_smith_set == {0, 1, 2}
    assert hare_obj.hare() == 0
    assert hare_obj.benham_hare() == 0
    assert hare_obj.smith_hare() == 0
    assert hare_obj.tideman_hare() == 0
    assert hare_obj.woodall_hare() == 0

