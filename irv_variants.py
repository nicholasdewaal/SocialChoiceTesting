from collections import defaultdict
from ipdb import set_trace
import tarjan as tr
import svvamp


# def copy_list_of_lists(in_list):  # much faster than deepcopy
    # out_list = [y.copy() for y in in_list]
    # return out_list


def get_num_votes_i_over_j(pref_ballots): # duplicate code, remove eventually!!
    '''
    num_votes_i_over_j[i][j] = # of votes for candidate i over candidate j
    '''
    N = len(pref_ballots[0])
    num_votes_i_over_j = N * [N * [0]]
    for ballot in pref_ballots:
        for kk, c_pref in enumerate(ballot):
            for c_less_pref in ballot[kk+1:]:
                num_votes_i_over_j[c_pref][c_less_pref] += 1
    return num_votes_i_over_j


class IRV_Variants():

    def __init__(self, all_ballots, num_i_to_j=None):
        self._all_ballots = all_ballots
        self._n_candidates = len(all_ballots[0])
        self._candidates = set(all_ballots[0])
        # _n_vote_i_to_j[i][j] = pct or # of votes for candidate i over j
        self._n_vote_i_to_j = num_i_to_j if num_i_to_j else \
            get_num_votes_i_over_j(self._all_ballots)
        self._primary_smith_set = self.get_smith_set(self._candidates)
        # self._pref_graph =
        # to do: get the preference graph once, then get the Smith
        # set for a given subset fewer times.


    def get_smith_set(self, candidates_to_check):
        '''
        returns tuple of cycle elements
        '''
        graph = defaultdict(list)
        for i in candidates_to_check:
            for j in candidates_to_check:
                # Candidate i domminates Candidate j
                if i != j and self._n_vote_i_to_j[i][j] >= self._n_vote_i_to_j[j][i]:
                    graph[i].append(j)
        cycles = set(tuple(x) for x in tr.tarjan(graph))
        removal_set = set()
        # if a member of cycle x is preferred to a member of cycle y, then all of
        # cycle x is preferred to y, so y is not the smith set.
        for x in cycles:
            for y in cycles:
                if x != y and y[0] in graph[x[0]]:
                    removal_set.add(y)
        smith_set = set((cycles - removal_set).pop())
        return smith_set


    def find_nxt_loser(self, existing_losers):
        counts = defaultdict(int)
        for ballot in self._all_ballots:
            for val in ballot:
                if val not in existing_losers:
                    break
            counts[val] += 1

        # Find a winner if there is one
        winner = None
        n_ballots = len(self._all_ballots)
        for k, v in counts.items():
            if v / n_ballots > .5:
                winner = k
                break

        # Find Candidates with the worst top vote counts, add to losers
        min_candidate = min(counts, key=counts.get)
        min_counts = counts[min_candidate]
        losers = set(existing_losers)
        for candidate, value in counts.items():
            if value == min_counts:
                losers.add(candidate)

        return winner, losers


    def hare(self, losers=set()):
        '''
        Classic Instant Runoff Voting
        '''
        winner = None
        while not winner:
            winner, losers = self.find_nxt_loser(losers)
        return winner


    def benham_hare(self):
        '''
        SAME AS ICRV IN svvamp
        Eliminate Plurality loser until a Condorcet winner exists.
        '''
        losers = set()
        while True:
            smith_set = self.get_smith_set(self._candidates - losers)
            if len(smith_set) == 1:
                return smith_set.pop()
            _, losers = self.find_nxt_loser(losers)


    def woodall_hare(self):
        '''
        get irv ranks
        choose rank best in smith set as winner
        '''
        if len(self._primary_smith_set) == 1:
            return list(self._primary_smith_set)[0]

        irv_rank, irv_set, losers, prev_losers = list(), set(), set(), set()

        while not self._primary_smith_set.issubset(irv_set):
        # while len(irv_rank) < self._n_candidates:
            _, losers, prev_losers = *self.find_nxt_loser(losers), losers
            for x in losers - prev_losers:
                irv_rank.insert(0, x)
                irv_set.add(x)

        for x in irv_rank:
            if x in self._primary_smith_set:
                return x


    def smith_hare(self):
        '''
        Eliminate candidates not in the Smith set, then run hare on the remaining.
        '''
        if len(self._primary_smith_set) == 1:
            return list(self._primary_smith_set)[0]
        losers = self._candidates - self._primary_smith_set
        _, losers = self.find_nxt_loser(losers)
        winner = self.hare(losers=losers)
        return winner


    def tideman_hare(self):
        '''
        Alternate between eliminating candidates not in the smith set, and the
        plurality loser.
        '''
        winner, losers = None, set()
        while not winner:
            remaining_candidates = self._candidates - losers
            smith_set = self.get_smith_set(remaining_candidates)
            if len(smith_set) == 1:
                return smith_set.pop()
            else:
                losers = losers.union(self._candidates - smith_set)
                winner, losers = self.find_nxt_loser(losers)
        return winner

