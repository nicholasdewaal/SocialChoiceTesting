
from pref_matrix.pref_matrix import c_gen_pref_summaries
import irv_variants as irv
import election_random as er
import pytest
from numpy import array, intc
import svvamp
# to do: fix imports, create packages, __init__ files


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
    counts, _ = er.gen_pref_summaries(pref_ballots)
    w = er.get_weights_from_counts(counts)
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
        pop = svvamp.PopulationSpheroid(V=100*n, C=n)
        pref_ballots = pop.preferences_rk.tolist()
        p = array(pref_ballots, dtype=intc)
        r1, r2 = c_gen_pref_summaries(p)
        t1, t2 = er.gen_pref_summaries(pref_ballots)
        assert (t1 == r1).all()
        assert (t2 == r2).all()


def test_gen_weights_zipf():

    for n in [5, 10, 100, 1000]:
        x = er.gen_weights_zipf(n)
        assert len(x) == n
        assert abs(sum(x) - 1) < .0001
    # add test for distribution?


def test_gen_ranked_preferences_zipf():

    n_candidates = 5
    ordered_ballot = list(range(n_candidates))
    for n_voters in [5, 15, 200, 1000]:
        ballots = er.gen_ranked_preferences_zipf(n_candidates, n_voters)
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
    u = er.social_util_by_cand(ranked_weights)
    assert u[4] == 1
    assert u[0] > u[1]
    ranked_weights = [[0.1, 0.9],
                      [0.9, 0.1]]
    u = er.social_util_by_cand(ranked_weights)
    assert u[0] < u[1]
    assert u[1] == 1

    ranked_weights = [[1.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.0, 0.5, 0.5]]
    u = er.social_util_by_cand(ranked_weights)
    assert u == {0: 1.0, 1: 0.375, 2: 0.375}
    u = er.social_util_by_cand(ranked_weights, .999999999)
    for v in u.values():
        assert abs(v - 1) < .001
    u = er.social_util_by_cand(ranked_weights, 0.00000001)
    assert u[0] == 1
    assert u[1] < .0001
    assert u[2] < .0001

    ranked_weights = [[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]
    for v in [0.1, 0.2, .3, .4, .5, .6, .7, .8, .9]:
        u = er.social_util_by_cand(ranked_weights, v)
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
        assert er.get_pairoff_winner({i, i+1}, pref_ij) == i

    assert er.get_pairoff_winner({0, 4}, pref_ij) == 0
    # not a legitimate pref matrix, but good enough for tests
    pref_ij = [[5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1]]
    for i in range(4):
        assert er.get_pairoff_winner({i, i+1}, pref_ij) == i+1

    assert er.get_pairoff_winner({0, 3}, pref_ij) == 3


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
    if pop_iter == er.iter_rand_pop_zipf:
        it, _ = pop_iter(n_voters, n_candidates)
    else:
        it, _ = pop_iter(n_voters, n_candidates, param)
    num_items = 0
    for x in it:
        num_items += 1
        prefs = x.preferences_rk.tolist()
        assert len(prefs) == n_voters
        assert len(prefs[0]) == n_candidates
        assert x.C == n_candidates
        assert x.V == n_voters

    if pop_iter == er.iter_rand_pop_zipf:
        pass

    elif pop_iter == er.iter_rand_pop_gauss:
        assert num_items == num_param * 3
    else:
        assert num_items == param

def test_iter_rand_pop_polar():
    standard_iter_tester(er.iter_rand_pop_polar)


def test_iter_rand_pop_other():
    standard_iter_tester(er.iter_rand_pop_other)


def test_iter_rand_pop_zipf():
    standard_iter_tester(er.iter_rand_pop_zipf)


def test_iter_rand_pop_gauss():
    standard_iter_tester(er.iter_rand_pop_gauss)


def test_iter_rand_pop_ladder():
    standard_iter_tester(er.iter_rand_pop_ladder)


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

