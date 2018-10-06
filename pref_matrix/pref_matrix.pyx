
import numpy as np

# input: p=np.array(pop.preferences_rk, dtype=np.intc)

def c_gen_pref_summaries(int[:, :] pref_ballots):
    cdef int n = pref_ballots.shape[0]
    cdef int m = pref_ballots.shape[1]
    cdef int[:, :] n_pref_i_over_j = np.zeros((n, n), dtype=np.intc)
    cdef int[:, :] n_pref_by_rank = np.zeros((n, n), dtype=np.intc)

    cdef int i, j, k, d

    for i in range(n): # iterate over ballots
        for j in range(m): #iterate over ranks
            n_pref_by_rank[j, pref_ballots[i, j]] += 1
            for k in range(j+1, m):
                n_pref_i_over_j[pref_ballots[i, j], pref_ballots[i, k]] += 1

    return n_pref_by_rank, n_pref_i_over_j
