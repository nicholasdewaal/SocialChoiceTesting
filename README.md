# SocialChoiceTesting

  Although the main purpose of this software is to test and analyse the results of various social-choice functions, it can also be used to run some hard-to find implementations of important social-choice functions such as:

Ballots must be complete rank-order ballots of candidates numbered 0-n as a list of lists, tuples of tuples, numpy array, or something similarly compatible.
  The random ballot:
  ```python
  from lottery_scfs import random_ballot, multi_lottery
  from ballot_generators import gen_ranked_preferences_zipf

  pref_ballots = gen_ranked_preferences_zipf(n_candidates=5, n_voters=5000,
                                             zipf_param=1.5)
  winner = random_ballot(pref_ballots)
  ```
  Various variations of multi-lottery style elections:

  ```python
  winner = multi_lottery(pref_ballots, points_to_win=5, method='borda_decay')
  winner = multi_lottery(pref_ballots, points_to_win=7, method='borda')
  winner = multi_lottery(pref_ballots, points_to_win=20, method='plurality')
  winner = multi_lottery(pref_ballots, points_to_win=5, method='iterated_borda')
  ```

  Variants on Instant runoff such as Benham-Hare, Smith-Hare, Tideman-Hare, and Woodall Hare:

  ```python
  import irv_variants as irv

  hare_obj = irv.IRV_Variants(pref_ballots)
  winner = hare_obj.benham_hare()
  winner = hare_obj.smith_hare()
  winner = hare_obj.tideman_hare()
  winner = hare_obj.woodall_hare()
  ```

  This software simulates several prominent social-choice functions with numerous randomly-generated samples of different types of voting populations to learn which vote-tallying techniques statistically tend to have the best desired results for democratic processes. This software includes some of my ideas for new and hopefully better social-choice functions.

  The overall goal of this project is to find a social-choice function that better addresses constraints of election manipulability according to Gibbard's Theorem, and constraints from Arrow's Theorem. Another important consideration is the design of election systems that would maintain the stability and feasibility of governments, and maintain democracy in societies with highly polarized politics. Selecting a social-welfare function that optimizes social welfare relative to such constraints is the overall goal.

  This software is a work in progress, and some parts may still need refining.

  This code runs with Python 3 only.  It is not compatible with Python 2.\*

  All simulations assume results based upon completely sincere voting which is never true in theory except for the random ballot. Most social-choice functions are easier than others to manipulate: https://www.rangevoting.org/mani-focs.pdf

  The purpose of this simulation is to compare social-choice functions with must better resistance to manipulation to more popular functions under ideal circumstances to consider how much potential loss comes from decreasing manipulability.

  You can run this simulation by navigating to the folder with the file election_random.py, and executing:

  "python3 -O simulated_elections.py"

  If you want to save the text output from the simulation, run:

  "python3 -O simulated_elections.py &>> summary.txt"

The license for using this software is GPL v3 as written in the included file License.txt.

