
from pref_matrix.pref_matrix import c_gen_pref_summaries
import irv_variants as irv
import election_random as er
import pytest
from numpy import array, intc
# to do: fix imports


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
    counts, _ = gen_pref_summaries(pref_ballots)
    w = get_weights_from_counts(counts)
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

    assert wt = expected


def test_gen_pref_summaries():
    for n in range(5):
        pop = svvamp.PopulationSpheroid(V=100*n, C=n)
        pref_ballots = pop.preferences_rk.tolist()
        p = array(pref_ballots, dtype=intc)
        r1, r2 = c_gen_pref_summaries(np_counts)
        t1, t2 = gen_pref_summaries(pref_ballots)
        assert t1 == r1
        assert t2 == r2


def test_gen_weights_zipf():

    for n in [5, 10, 100, 1000]:
        x = gen_weights_zipf(n)
        assert len(x) == n
        assert abs(sum(x) - 1) < .0001
    # add test for distribution?


def test_gen_ranked_preferences_zipf():

    for n_candidates in [5, 15, 200, 1000]:
        ballots = gen_ranked_preferences_zipf(n_candidates, 5)
        assert len(ballots) == n_candidates
        assert len(ballots[0]) == n_voters
        for b in ballots:
            assert sorted(b) == [1, 2, 3, 4, 5]
    # add test for distribution?


def test_social_util_by_cand():
    ranked_weights =
    social_util_by_cand(ranked_weights)

