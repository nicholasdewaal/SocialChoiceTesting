from collections import defaultdict
from ipdb import set_trace
import tarjan as tr
from pref_matrix.pref_matrix import c_gen_pref_summaries
from numpy import intc, array


# def copy_list_of_lists(in_list):  # much faster than deepcopy, not used now
#     return [y.copy() for y in in_list]

def verify_ballots_legitimate(all_ballots):
    ball_vals = set(range(len(all_ballots[0])))
    for ballot in all_ballots:
        if set(ballot) != ball_vals:
            return False
    return True


class IRV_Variants():
    '''
    These social-choice functions were developed according to the methods
    described here: http://www.votingmatters.org.uk/ISSUE29/I29P1.pdf
    '''

    def __init__(self, all_ballots, num_i_to_j=None):
        '''
        all_ballots is a list of ballots, each of which is a list of candidate
        rankings.
        Ballots are only legitimate if each candidate is numbered 0 to n - 1
        where n is the number of candidates. Each ballot must have a full
        ranking of the candidates with the index in the list being the rank of
        the candidate for a given ballot.
        '''
        assert max(max(x) for x in all_ballots) == len(all_ballots[0]) - 1
        assert verify_ballots_legitimate(all_ballots)
        self._all_ballots = all_ballots
        self._n_candidates = len(all_ballots[0])
        self._candidates = set(all_ballots[0])
        # _n_vote_i_to_j[i][j] = pct or # of votes for candidate i over j
        if num_i_to_j is None:
            pref_ballots = array(self._all_ballots, dtype=intc)
            self._n_vote_i_to_j = c_gen_pref_summaries(pref_ballots)[1]
        else:
            self._n_vote_i_to_j = num_i_to_j

        self._primary_smith_set = self.get_smith_set(self._candidates)

    def get_smith_set(self, candidates_to_check):
        '''
        This function returns a tuple of cycle elements
        The Smith set is the largest set of candidates where no other
        candidates outside such a set are preferred over those inside the set.
        If the size of the Smith set is 1, then that is the Condorcet winner.
        Otherwise there is no Condorcet winner.

        '''
        if len(candidates_to_check) == 1:
            return set(candidates_to_check)

        graph = defaultdict(list)
        for i in candidates_to_check:
            for j in candidates_to_check:
                # Candidate i domminates Candidate j
                if i != j and self._n_vote_i_to_j[i][j] \
                        >= self._n_vote_i_to_j[j][i]:
                    graph[i].append(j)
        cycles = set(tuple(x) for x in tr.tarjan(graph))
        removal_set = set()
        # if a member of cycle x is preferred to a member of cycle y, then all
        # of cycle x is preferred to y, so y is not the smith set.
        for x in cycles:
            for y in cycles:
                if x != y and y[0] in graph[x[0]]:
                    removal_set.add(y)
        smith_set = set((cycles - removal_set).pop())
        return smith_set

    def find_nxt_loser(self, existing_losers):
        '''
        Eliminate the next loser as the candidate with the fewest 1st-choice
        votes that aren't already in the set of existing_losers.
        Every candidate with the number of 1st-choice votes equal to the fewest
        is added to the existing_losers set and returned as the loser set.

        If there is a winner with over 50% of the 1st-choice votes, then the
        winner is also returned.
        If all candidates are losers by an all-way tie, then a miscellaneous
        winner is chosen.

        '''

        counts = {i: 0 for i in set(self._all_ballots[0]) - existing_losers}
        # if a candidate is in existing_losers, don't add to count since they
        # have already been eliminated. Go to the next candidate on the ballot
        # until you find a non-loser to augment their vote count.
        for ballot in self._all_ballots:
            for val in ballot:
                if val not in existing_losers:
                    break
            counts[val] += 1

        # Find a winner if there is one that has more than 50% of the votes
        winner = None
        n_ballots = len(self._all_ballots)
        for k, v in counts.items():
            if v / n_ballots > .5:
                winner = k
                break

        # Find candidates with the worst top vote counts, add to losers
        min_candidate = min(counts, key=counts.get)
        min_counts = counts[min_candidate]
        losers = set(existing_losers)
        for candidate, value in counts.items():
            if value == min_counts:
                losers.add(candidate)
        if losers == self._candidates:
            winner = losers.pop() # not all can be losers, pick any one then

        return winner, losers

    def hare(self, losers=set()):
        '''
        Classic Instant Runoff Voting
        '''
        winner = None
        while winner == None:
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
        choose rank best in Smith set as winner
        '''
        if len(self._primary_smith_set) == 1:
            return list(self._primary_smith_set)[0]

        irv_rank, irv_set, losers, prev_losers = list(), set(), set(), set()

        while not self._primary_smith_set.issubset(irv_set):
            winner, losers, prev_losers = *self.find_nxt_loser(losers), losers
            if losers==prev_losers and winner not in irv_set:
                irv_rank.insert(0, winner)
                irv_set.add(winner)
            for x in losers - prev_losers:
                irv_rank.insert(0, x)
                irv_set.add(x)

        for x in irv_rank:
            if x in self._primary_smith_set:
                return x

    def smith_hare(self):
        '''
        Eliminate candidates not in the Smith set, then run hare on those
        remaining.
        '''
        if len(self._primary_smith_set) == 1:
            return list(self._primary_smith_set)[0]
        losers = self._candidates - self._primary_smith_set
        _, losers = self.find_nxt_loser(losers)
        winner = self.hare(losers=losers)
        return winner

    def tideman_hare(self):
        '''
        Alternate between eliminating candidates not in the Smith set, and the
        plurality loser.
        '''
        winner, losers = None, set()
        while winner == None:
            remaining_candidates = self._candidates - losers
            smith_set = self.get_smith_set(remaining_candidates)
            if len(smith_set) == 1:
                return smith_set.pop()
            else:
                losers = losers.union(self._candidates - smith_set)
                winner, losers = self.find_nxt_loser(losers)
        return winner
