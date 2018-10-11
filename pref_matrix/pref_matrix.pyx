

from numpy import zeros, intc, asarray


def c_gen_pref_summaries(int[:, :] pref_ballots):
    '''
    input: pref_ballots = array(pop.preferences_rk, dtype=intc)
    '''
    cdef int n_voters = pref_ballots.shape[0]
    cdef int n_cand = pref_ballots.shape[1]
    cdef int[:, :] n_pref_i_over_j = zeros((n_cand, n_cand), dtype=intc)
    cdef int[:, :] n_pref_by_rank = zeros((n_cand, n_cand), dtype=intc)

    cdef int i, j, k, d

    for i in range(n_voters): # iterate over ballots
        for j in range(n_cand): #iterate over ranks
            n_pref_by_rank[j, pref_ballots[i, j]] += 1
            for k in range(j + 1, n_cand):
                n_pref_i_over_j[pref_ballots[i, j], pref_ballots[i, k]] += 1

    return asarray(n_pref_by_rank), asarray(n_pref_i_over_j)
