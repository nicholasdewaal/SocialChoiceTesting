#!/usr/bin/env python3

from addict import Dict
from numpy import arange, array, intc
from collections import defaultdict
import matplotlib.pyplot as plt
# from ipdb import set_trace
import svvamp
import irv_variants
import ballot_generators as bg
import lottery_scfs as ls
from pref_matrix.pref_matrix import c_gen_pref_summaries
from pandas import DataFrame
from math import ceil
import os
from shutil import move


def get_opt_dict_key(in_dict, op=max):
    return op(in_dict, key=in_dict.get)


def scale_utilities(in_utilities):  # not used?
    scaled_util = [(x - min(x)) / (max(x) - min(x)) for x in in_utilities]
    # Scaled to get a percent total satisfaction of a population.
    # Average happiness by candidate
    return sum(scaled_util) / len(scaled_util)


def social_util_by_cand(ranked_weights, fraction_happy_decay=.25):
    '''
    Assume a multiplicative fractional utility decay for a voter
    by each drop in their preference ranking
    '''
    ls.assert_weights_sound(ranked_weights)
    assert fraction_happy_decay < 1 and fraction_happy_decay > 0

    # we want the average preferred candidate to be fraction_happy_decay
    # as desirable as the best choice. Solve for fraction_happy_decay:
    # decay_rate ** ((L-1) / 2) = fraction_happy_decay

    decay_rate = fraction_happy_decay ** (2 / (len(ranked_weights) - 1))
    happiness = defaultdict(int)

    # loop over top to bottom choice
    for ii, weights in enumerate(ranked_weights):
        # loop over candidates
        for jj, candidate_val in enumerate(weights):
            happiness[jj] += candidate_val * decay_rate ** ii
    max_happiness = max(happiness.values())
    # rescale to 1 the best happiness
    for x in happiness:
        happiness[x] = happiness[x] / max_happiness
    return happiness


def simulate_multi_lottery(pref_ballots, social_happinesses, n_pref_by_rank,
                           pref_ij, num_sim_per_cand=1000, n_pts_win=2,
                           method='borda', recursive=True):
    '''
    This simulates selecting the winner of a given election with voting
    ballots = pref_ballots using an improved variation on the random ballot
    called multi_lottery.

    The simulation is repeated num_sim_per_cand times the number of candidates
    times.
    '''

    n_candidates = len(pref_ballots[0])
    # More simulations per candidate helps to improve statistical accuracy.
    num_sim = num_sim_per_cand * n_candidates
    current_sim = 0
    num_primaries_won = {c: 0 for c in range(n_candidates)}
    num_finals_won =  {c: 0 for c in range(n_candidates)}
    happiness_freqs = list()

    while current_sim < num_sim:  # Do num_sim simulated elections
        # Handle simulated primary elections
        primary_winners, finals_winner = ls.multi_lottery(pref_ballots,
                        n_pts_win, borda_decay=.5, pref_ij=pref_ij,
                        n_pref_by_rank=n_pref_by_rank, method=method)
        for winner in primary_winners:
            num_primaries_won[winner] += 1
        # Handle simulated final elections
        num_finals_won[finals_winner] += 1
        current_sim += 1

    # Since 2 candidates win each primary, the frequecy sums to 2, not 1
    freq_primry_won = {key: float(val) / num_sim for key, val in
                       num_primaries_won.items()}
    freq_finals_won = {key: float(val) / num_sim for key, val in
                       num_finals_won.items()}

    # In the following, happiness_freqs is a list of tuples of social happiness
    # occurring in the election simulation at a frequency corresponding to
    # the second element of each tuple: (happiness_measure, frequency in sim)

    for candidate_key, happy_ms in social_happinesses.items():
        happiness_freqs.append((happy_ms, freq_finals_won[candidate_key]))

    # The following is the average social happiness from the election results
    # in the simulation.
    avg_happiness = sum(h[0] * h[1] for h in happiness_freqs)

    # If someone wins mostly from a loser category, debug to see why.
    p = n_candidates // 4  # Lowest quartile
    lower_pctls = array(sorted(social_happinesses, key=social_happinesses.get),
                        dtype=int)[:p]
    highest_winner = get_opt_dict_key(freq_finals_won, max)
    if n_pts_win >= 2 and highest_winner in lower_pctls and recursive:
        print('Why is lower pctl candidate winning so much! Debug!')
        freq = simulate_multi_lottery(pref_ballots, social_happinesses,
                    n_pref_by_rank, pref_ij, num_sim_per_cand,
                    n_pts_win, method, False)[1]
        with open('Weird_results.txt', 'a') as f:
            f.write('\n\nFinal winner, points to win: ' + str(n_pts_win))
            f.write('\nCand: Freq, Socl Happiness:\n')
            for k, v in freq.items():
                f.write('%d: %.4f,  %.4f\n' % (k, v, social_happinesses[k]))
        # set_trace()

    return freq_primry_won, freq_finals_won, happiness_freqs, avg_happiness


def print_winnerpct_dict(in_dict):
    for x, y in in_dict.items():
        print('candidate ', x, ': ', round(100 * y, 1), '%')


def mkdir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def plot_sim(pref_ballots, weights, n_pref_by_rank, pref_ij, dir_name,
             parent_folder, test_point_cuttoffs=[1, 1.1, 1.9, 2, 2.2, 2.9, 3],
             method='borda'):

    '''
    Given a particular set of ballots: pref_ballots, simulate tallying the
    votes numerous times using an improved variation on the random ballot to
    plot voter satisfaction averages and distributions.

    Plot the simulation of method='borda', or other
    multi_lottery method given point cutoffs of test_point_cuttoffs.
    Plots include distributions of happiness for multiple simulations of
    the same election, and frequencies of candidates surviving the primary
    election. Plots are saved in the folder dir_name.
    '''
    ls.assert_weights_sound(weights)
    n_candidates = len(weights)
    dir_path = parent_folder + '/' + dir_name
    mkdir_if_not_exist(dir_path)

    fig, ax = plt.subplots()
    # fig.patch.set_facecolor('xkcd:gray')
    # ax.set_facecolor((0.38, 0.34, 0.22))
    bar_width = 0.75 / len(test_point_cuttoffs)
    opacity = 0.8
    colors = 'kbgrcmy'
    num_sim_per_cand = 1500
    freq_history = defaultdict(float)

    social_happiness = social_util_by_cand(weights)
    index = array(range(n_candidates))
    s_index = array(sorted(social_happiness, key=social_happiness.get,
                         reverse=True), dtype=int)

    for j, pts in enumerate(test_point_cuttoffs):

        freq_primry_won, freq_finals_won, happiness_freqs, avg_happiness = \
            simulate_multi_lottery(pref_ballots, social_happiness,
                                   n_pref_by_rank, pref_ij, num_sim_per_cand,
                                   n_pts_win=pts, method=method)
        print('happiness order:', s_index)
        print(social_happiness)
        freq_history[pts] = [happiness_freqs, avg_happiness]
        min_key = get_opt_dict_key(social_happiness, min)
        min_val = round(100 * social_happiness[min_key], 1)
        max_key = get_opt_dict_key(social_happiness, max)
        max_val = round(100 * social_happiness[max_key], 1)
        print('\nFor', pts, 'points to win primary, avg_happiness =',
              round(100 * avg_happiness, 1), '%, ', dir_name)
        print('Worst social happiness is candidate %d =' % min_key, min_val, '%')
        print('Best social happiness is candidate %d =' % max_key, max_val, '%')
        # print("happiness_distr: ", happiness_freqs)
        print('Final_winner percentages won in simulation: ')
        print_winnerpct_dict(freq_finals_won)

        primary_freqs = [100 * freq_primry_won[ii] for ii in s_index]
        finals_freqs = [100 * freq_finals_won[ii] for ii in s_index]
        plt.subplot(2, 1, 1)
        plt.bar(index + j * bar_width, primary_freqs, bar_width,
                alpha=opacity, color=colors[(j + 1) % len(colors)])#,
                # label=str(pts) + ' points')
        plt.xticks(s_index + bar_width, [str(x) for x in range(n_candidates)])
        plt.ylabel('% Winning Primary\n(2 winners)')
        plt.subplot(2, 1, 2)
        plt.bar(index + j * bar_width, finals_freqs, bar_width,
                alpha=opacity, color=colors[(j + 1) % len(colors)],
                label=str(pts) + ' points')

    plt.xlabel('Candidate')
    plt.ylabel('% Winning Finals')
    plt.suptitle('Simulated wins by candidates 0-' + \
                 '%d using multi-lottery method='%(n_candidates - 1) + \
                 method + ' for each p points to win' + \
                 ' in order from most preferred to least.',
                 fontsize=6)
    plt.xticks(s_index + bar_width, [str(x) for x in range(n_candidates)])
    plt.legend(loc='best', fontsize=5)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(dir_path + '/Percent_of_time_win_primaries_' +
                str(n_candidates) + '_candidates.png', dpi=200)

    plt.gcf().clear()

    n = len(test_point_cuttoffs)
    plot_num = 1
    num_columns = ceil(n ** .5)
    num_rows = ceil(n / num_columns)
    ordered_keys = sorted(freq_history.keys())

    for pt_lim in ordered_keys:
        happy_data = freq_history[pt_lim]
        plt.subplot(num_rows, num_columns, plot_num)
        s = list(zip(*happy_data[0]))
        tmp = sorted(s[0])
        min_dist = 1 if len(tmp) == 1 else \
            min(tmp[i + 1] - tmp[i] for i in range(len(tmp) - 1))
        plt.xlabel(str(pt_lim) + ' point threshold', fontsize=7)
        plt.bar(s[0], s[1], width=max(min_dist * .9, .0038), )
        plt.bar([happy_data[1]], [1], width=max(min_dist * .35, .003),
                color='r')
        plt.tick_params(axis='both', which='major', labelsize=5)
        plt.tick_params(axis='both', which='minor', labelsize=5)
        plot_num += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Happiness (0-1) frequencies by point threshold.' +
                 ' Red is average happiness.')
    plt.savefig(dir_path + '/Happiness_frequencies_final_winner_sim_' +
                str(n_candidates) + '_candidates.png', dpi=200)

    fig.clf()
    plt.close()


def simulate_all_elections(pop_object, fast=False, pref_i_to_j=None,
                           n_pref_by_rank=None):
    '''
    fast = True: This eliminates time consuming SCF's from testing which
    includes range voting and Baldwin. Tactical voting for range voting is
    essentially approval voting which is simulated, and Nanson is extremely
    similar to Baldwin.

    Borda-IRV: Baldwin, Nanson

    Borda: classic
    Condorcet: classic
    that of all other candidates

    Plurality/FPTP

    Implement Later: Dowdall, Symmetric borda, combine Baldwin
    (borda-irv) with Condorcet smith? Copeland
    Simpson: choose the candidate whose worst pairwise defeat is better than
    Raynaud (remove biggest Condorcet loser # iteratively)

    Multi-optimization objectives for choosing a social-choice function
    should be to:
       1. Minimize election manipulability from candidates, voters or
          coalitions.
       2. Maximize social welfare/utility.
       3. Maximize political stability.

    Sometimes social welfare is less important than maintaining a stable
    government. Highly polarized societies may cause an unstable government if
    a Condorcet winner is chosen; those on the losing polar extreme may work
    hard to undermine government, or cause civil war/unrest. Instead, consensus
    candidates improve stability.

    Condorcet methods in polarized societies cause wolves to eat the sheep in
    majority dominance. Borda-style SCF's cause a more consensus candidate to
    win. Iraqi Shia vs. Sunnis, or US politics from 2012-now are good examples.

    '''
    pref_ballots = pop_object.preferences_rk.tolist()
    if not(n_pref_by_rank and pref_i_to_j):
        n_pref_by_rank, pref_i_to_j = ls.fast_gen_pref_summ(pref_ballots)
    results = dict()  # name each election type
    hare_obj = irv_variants.IRV_Variants(pref_ballots, num_i_to_j=pref_i_to_j)
    results['tideman_hare'] = hare_obj.tideman_hare()
    results['smith_hare'] = hare_obj.smith_hare()
    results['woodall_hare'] = hare_obj.woodall_hare()
    results['benham_hare'] = svvamp.ICRV(pop_object).w
    results['hare'] = svvamp.IRV(pop_object).w

    results['schulze'] = svvamp.Schulze(pop_object).w
    results['borda'] = svvamp.Borda(pop_object).w
    results['nanson'] = svvamp.Nanson(pop_object).w  # borda-irv below avg
    if not fast:
        results['baldwin'] = svvamp.Baldwin(pop_object).w  # borda-irv
        results['range'] = svvamp.RangeVotingAverage(pop_object).w
    results['irv-duels'] = svvamp.IRVDuels(pop_object).w
    results['approval'] = svvamp.Approval(pop_object).w
    results['plurality'] = svvamp.Plurality(pop_object).w
    # Like IRV, but elim most low rank
    results['coombs'] = svvamp.Coombs(pop_object).w
    # multi_lottery method simulated 1 times, in sim, average will show in end.

    results['random_ballot'] = ls.random_ballot(pref_ballots)
    _, results['lottery_borda2'] = ls.multi_lottery(pref_ballots, 2,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_bor2.3'] = ls.multi_lottery(pref_ballots, 2.3,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_borda3'] = ls.multi_lottery(pref_ballots, 3,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_bor3.8'] = ls.multi_lottery(pref_ballots, 3.8,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_borda5'] = ls.multi_lottery(pref_ballots, 5,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_bord12'] = ls.multi_lottery(pref_ballots, 12,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_bord50'] = ls.multi_lottery(pref_ballots, 50,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='borda')
    _, results['lottery_plura2'] = ls.multi_lottery(pref_ballots, 2,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='plurality')
    _, results['lottery_plura5'] = ls.multi_lottery(pref_ballots, 5,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='plurality')
    _, results['lottery_plur15'] = ls.multi_lottery(pref_ballots, 15,
        pref_ij=pref_i_to_j, n_pref_by_rank=n_pref_by_rank, method='plurality')

    return results


def get_happinesses_by_method(pop_iterator, fast=False):

    num_sim, current_sim = 1500, 0
    utils_by_scf = Dict()
    dataframe_dict = Dict()
    test_num_candidates = [3, 4, 6, 9, 13, 18, 24]

    # modify each sim to run in parallel
    while current_sim < num_sim:
        print(current_sim)
        # simulate for various numbers of candidates
        for n_candidates in test_num_candidates:
            n_voters = n_candidates * 750

            for pop, param in pop_iterator(n_voters, n_candidates):
                n_pref_by_rk, pref_ij = ls.fast_gen_pref_summ(pop.preferences_rk)
                weights = ls.get_weights_from_counts(n_pref_by_rk)
                utils = social_util_by_cand(weights)
                winners_by_scf = simulate_all_elections(pop, fast=fast,
                    n_pref_by_rank=n_pref_by_rk, pref_i_to_j=pref_ij)

                utils_by_scf[param][n_candidates][current_sim] = \
                    {k: utils[v] for k, v in winners_by_scf.items()}
        current_sim += 1
    save_directory = 'Population_type_sim=' + pop_iterator.__name__
    archive_old_sims(save_directory, 'Previous_sims_all_methods')
    # utils_by_scf[pop_param][n_candidates][sim_number][scf]
    # now make dict of DataFrames by paramaters, n_candidates
    os.mkdir(save_directory)
    for param, v_upper in utils_by_scf.items():
        for n_cand, scf_by_sim_num in v_upper.items():
            dataframe_dict[param][n_cand] = DataFrame.from_dict(scf_by_sim_num,
                                                                orient='index')
            dataframe_dict[param][n_cand].boxplot(rot=90)  # labels? by axis?
            plt.tight_layout()
            plt.savefig(save_directory + '/plot_p=' + str(param) +
                        '_n_cand=' + str(n_cand) + '.png')
            plt.close()


def archive_old_sims(old_sim_subname, new_folder_name):
    # make new folder for saving old sims
    done = False
    num_attempts = 1
    contents = os.listdir()
    if any([True for x in contents if x.find(old_sim_subname) != -1]):
        while not done:
            try:
                new_folder = new_folder_name + ' (' + str(num_attempts) + ')'
                os.mkdir(new_folder)
                for val in contents:
                    if val.find(old_sim_subname) != -1:
                        move(val, new_folder)
                done = True
            except:
                num_attempts += 1


# fp=open('memory_profiler.log','w+')
# @profile(stream=fp)
def sim_with_iterator(pop_iterator, n_voters, n_cand, method, point_cuttoffs):
    parent_folder =  'method=' + method
    mkdir_if_not_exist(parent_folder)
    for pop, param in pop_iterator(n_voters, n_cand):
        votes = pop.preferences_rk
        n_pref_by_rank, pref_ij = ls.fast_gen_pref_summ(votes)
        w = ls.get_weights_from_counts(n_pref_by_rank)
        folder_name = 'data_gen=' + pop_iterator.__name__ + '_param=' + \
            str(param)
        plot_sim(votes, w, n_pref_by_rank, pref_ij, folder_name, parent_folder,
                    test_point_cuttoffs=point_cuttoffs, method=method)


def sim_single_elections(method='borda'):
    # Simulate multi_lottery and plot
    point_cuttoffs = [1, 1.5, 2, 2.1, 3, 3.5, 8, 20]
    n_voters = 5000

    for n in range(3, 11):
        sim_with_iterator(bg.iter_rand_pop_zipf, n_voters=n_voters, n_cand=n,
                        method=method, point_cuttoffs=point_cuttoffs)
        sim_with_iterator(bg.iter_rand_pop_polar, n_voters=n_voters, n_cand=n,
                        method=method, point_cuttoffs=point_cuttoffs)
        sim_with_iterator(bg.iter_rand_pop_gauss, n_voters=n_voters, n_cand=n,
                        method=method, point_cuttoffs=point_cuttoffs)
        sim_with_iterator(bg.iter_rand_pop_ladder, n_voters=n_voters, n_cand=n,
                        method=method, point_cuttoffs=point_cuttoffs)


def main1():
    archive_old_sims('method=', 'Previous_sims')
    sim_single_elections('borda')
    sim_single_elections('borda_decay')
    sim_single_elections('iterated_borda')
    sim_single_elections('iterated_borda_decay')
    sim_single_elections('plurality')


def main2():
    get_happinesses_by_method(bg.iter_rand_pop_polar, fast=True)
    get_happinesses_by_method(bg.iter_rand_pop_zipf, fast=True)
    get_happinesses_by_method(bg.iter_rand_pop_gauss, fast=True)
    get_happinesses_by_method(bg.iter_rand_pop_other, fast=True)
    get_happinesses_by_method(bg.iter_rand_pop_ladder, fast=True)


def test_sim():
    # Simulate all elections once
    pop = svvamp.PopulationVMFHypersphere(V=15000, C=15, vmf_concentration=2)
    res = simulate_all_elections(pop)
    s_keys = sorted(res.keys())
    # for x, y in res.items():
    max_len = len(max(s_keys, key=len))
    for k in s_keys:
        num_spc = max_len - len(k)
        print(k, ':', num_spc * ' ', res[k])


if __name__ == "__main__":
    main1()
    main2()
    # test_sim()
