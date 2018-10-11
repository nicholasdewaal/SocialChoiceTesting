
import irv_variants
import election_random
import pytest


def test_get_weights_from_counts(counts):
    w = get_weights_from_counts(counts)
    sums = [sum(b) for b in w]
    for s in sums:
        if abs(1-s) > .0001:
            return False

    Test cases:
    return True



def test_gen_pref_summaries(ballots):

    n_pref_by_rank, n_pref_i_over_j = gen_pref_summaries(ballots)



def test_gen_ranked_preferences_zipf(n_candidates, n_voters):

    ballots = gen_ranked_preferences_zipf(n_candidates, n_voters)
    if len(ballots) != n_candidates:
        return False
    if len(ballots[0]) != n_voters:
        return False
    Test cases:

    return True
