#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:27:26 2023

@author: ayumu
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io
import matplotlib as mp
from sklearn.decomposition import PCA
import random
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smt
import statsmodels.api as sm
import scipy.stats as stats
import glob
import os
from tqdm import tqdm
from collections import Counter
import sys
from joblib import Parallel, delayed


def compute_statistics_anova(pca_i, df_score_tmp, target_col):
    tmp = [df_score_tmp.query('%s==@cat' % target_col)[pca_i] for cat in np.unique(df_score_tmp[target_col])]
    tmp_stat, p_value = stats.f_oneway(*tmp)
    return tmp_stat, p_value


def pca_extract(df_score, target, method_pick_pca, fig_plot, fig_dir,
                bar_color='blue', n_pcs=None, alpha=0.05):
    """
    Identify PCs associated with the target variable.

    Parameters
    ----------
    df_score : pandas.DataFrame
        DataFrame containing only PCA scores (NaNs are assumed to be removed).
    target : pandas.Series or array-like
        Target variable (NaNs are assumed to be removed).
    method_pick_pca : str
        Multiple comparison correction method for selecting significant PCs.
        Used when n_pcs is None.
    fig_plot : bool
        Whether to generate and save plots of statistics.
    fig_dir : str
        Directory path to save plots. Required if fig_plot=True.
    bar_color : str, default='blue'
        Color for the bar plot.
    n_pcs : int or None, default=None
        If set, returns the top N PCs by absolute statistic.
        If None, uses multiple comparison correction.
    alpha : float, default=0.05
        Significance level for multiple comparison correction.

    Returns
    -------
    ind : numpy.ndarray
        Indices of selected PCs.
    statistics : numpy.ndarray
        Statistics for each PC.
    """
    # targetをSeriesに変換（まだSeriesでない場合）
    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    # df_scoreとtargetのインデックスを同期（既にNaNは除外されている想定）
    df_score_tmp = df_score.reset_index(drop=True)
    target_data = target.reset_index(drop=True)

    # データ型の判定
    is_numeric = pd.api.types.is_numeric_dtype(target_data)
    unique_vals = target_data.nunique()
    unique_list = sorted(target_data.unique())

    # target_nameの取得（プロットのタイトルやファイル名に使用）
    target_name = getattr(target, 'name', 'unknown')

    # binary（2値）の場合: ttest_ind
    if is_numeric and unique_vals == 2:
        val1, val2 = unique_list
        mask1 = target_data == val1
        mask2 = target_data == val2
        statistics, p = ttest_ind(
            df_score_tmp[mask1],
            df_score_tmp[mask2]
        )
    # 連続値の場合: pearsonr相関
    elif is_numeric and unique_vals > 2:
        results = Parallel(n_jobs=-1)(
            delayed(stats.pearsonr)(df_score_tmp[pca_i], target_data)
            for pca_i in df_score_tmp.columns
        )
        statistics, p = zip(*results)
    # カテゴリカルの場合: ANOVA
    else:
        df_score_anova = df_score_tmp.copy()
        target_col_name = target_name if target_name != 'unknown' else 'target'
        df_score_anova[target_col_name] = target_data
        # PCAカラムのみを処理（target_col_nameは除外）
        pca_columns = [col for col in df_score_anova.columns if col != target_col_name]
        results = Parallel(n_jobs=-1)(
            delayed(compute_statistics_anova)(pca_i, df_score_anova, target_col_name)
            for pca_i in tqdm(pca_columns)
        )
        statistics, p = zip(*results)

    if fig_plot:
        show_num = 20;
        plt.figure()
        plt.bar(range(len(statistics[:show_num])), np.abs(statistics[:show_num]), color=bar_color)
        plt.xticks(range(len(statistics[:show_num])), range(1, len(statistics[:show_num]) + 1))
        plt.title(target_name)
        plt.savefig(fig_dir + target_name + '.png')
        plt.savefig(fig_dir + target_name + '.svg')

    statistics = np.asarray(statistics)
    if n_pcs is not None:
        # Top-N by absolute statistic
        ind = np.argsort(np.abs(statistics))[::-1][:int(n_pcs)]
    else:
        # P-value based selection
        h = smt.multipletests(p, method=method_pick_pca, alpha=alpha)[0]
        ind = np.where(h)[0]
    print(f'{target_name} related pcs are {ind}')
    return ind, statistics


def con_extract(coeff, pcs, method):
    tmp_use_con = np.zeros(len(coeff))
    tmp_use_con_pc = np.zeros(len(coeff))
    if method == 'optimal_sigma':
        search_num = 5000
        for pc_i in pcs:
            cons, con_num_tmp = check_p_value(coeff, pc_i, search_num)
            tmp_use_con[cons] = 1
            tmp_use_con_pc[cons] = pc_i
    else:
        for pc_i in pcs:
            x = (coeff[:, pc_i] / np.std(coeff[:, pc_i])) ** 2
            p_values = 1 - stats.chi2.cdf(x, 1)
            h_final = smt.multipletests(p_values, method=method)[0]
            tmp_use_con[h_final] = 1
            tmp_use_con_pc[h_final] = pc_i

    cons = np.where(tmp_use_con)[0]
    cons_pc = tmp_use_con_pc[cons]
    con_num_tmp = len(cons)
    print(con_num_tmp)
    return cons, cons_pc


def check_p_value(coeff, ind, serch_num):
    use_coeff = coeff[:, ind]
    original_indices = np.arange(len(use_coeff))
    p_hist_val = []
    H = []
    removed_indices = []
    for i in tqdm(range(serch_num)):
        x = (use_coeff / np.std(use_coeff)) ** 2
        p_values = 1 - stats.chi2.cdf(x, 1)
        hist, bin_edges = np.histogram(p_values, bins=100)
        p_inf = hist / len(use_coeff)
        H.append(stats.entropy(p_inf))
        p_hist_val.append(np.var(hist))
        min_index = np.argmin(p_values)
        removed_indices.append(original_indices[min_index])
        use_coeff = np.delete(use_coeff, min_index)
        original_indices = np.delete(original_indices, min_index)
    con_num = np.argmin(p_hist_val)
    use_con = removed_indices[:con_num]
    return use_con, con_num


def select_optimal_pcs(
    target_pcs_lists,
    noise_pcs_lists,
    n_pcs=3,
    primary_target_pcs_lists=None
):
    """
    Select PCs with high target scores and low noise scores.

    Each input list contains (pc_index, score) tuples for one metric.
    Target and noise scores for each PC are the sum of absolute scores across
    the corresponding lists. The final score is target_score / noise_score.

    Parameters
    ----------
    target_pcs_lists : list of list of tuple
        List of (pc_index, score) lists for each target / supportive metric, e.g.
        [[(1, 4.5), (930, 3.2)], [(1, 3.8), (40, 2.9)]].
    noise_pcs_lists : list of list of tuple
        List of (pc_index, score) lists for each noise/confound metric, e.g.
        [[(2, 3.5), (4, 3.0)], [(10, 2.1), (35, 1.8)]].
    n_pcs : int, default=3
        Number of PCs to return.
    primary_target_pcs_lists : list of list of tuple or None, default=None
        If provided, selection is restricted to PCs that appear in these lists.
        These lists still contribute to the target score. Use this when you have
        a main target (e.g., diagnosis) and want the final PCs to be chosen from
        its top PCs, while supportive metrics (e.g., BDI) only modify the score.

    Returns
    -------
    selected_pcs : numpy.ndarray
        Indices of selected PCs, sorted by final score (best first).
    info : dict
        Dictionary containing scores and the input lists.
    """
    if not target_pcs_lists:
        raise ValueError("target_pcs_lists cannot be empty")

    if not noise_pcs_lists:
        raise ValueError("noise_pcs_lists cannot be empty")

    # Collect all PCs that appear in any list
    all_pcs = set()
    for pc_list in target_pcs_lists:
        for pc, _ in pc_list:
            all_pcs.add(pc)
    for pc_list in noise_pcs_lists:
        for pc, _ in pc_list:
            all_pcs.add(pc)

    # Determine candidate PCs: restrict to primary targets if provided,
    # otherwise allow any target PC.
    if primary_target_pcs_lists is not None and primary_target_pcs_lists:
        candidate_pcs_set = set()
        for pc_list in primary_target_pcs_lists:
            for pc, _ in pc_list:
                candidate_pcs_set.add(pc)
        all_pcs.update(candidate_pcs_set)
    else:
        candidate_pcs_set = all_pcs.copy()

    if not all_pcs:
        return np.array([], dtype=int), {
            'final_score': np.array([]),
            'target_score': np.array([]),
            'noise_score': np.array([]),
            'primary_target_pcs_lists': primary_target_pcs_lists,
            'target_pcs_lists': target_pcs_lists,
            'noise_pcs_lists': noise_pcs_lists,
        }

    n_pcs_total = max(all_pcs) + 1

    # Sum absolute scores for each PC.
    # target_pcs_lists contains all target metrics (primary + supportive).
    # primary_target_pcs_lists is only used for restricting the candidate set.
    target_score = np.zeros(n_pcs_total)
    for pc_list in target_pcs_lists:
        for pc, score in pc_list:
            target_score[pc] += abs(score)

    noise_score = np.zeros(n_pcs_total)
    for pc_list in noise_pcs_lists:
        for pc, score in pc_list:
            noise_score[pc] += abs(score)

    # Final score: high target, low noise
    epsilon = 1e-10
    final_score = target_score / (noise_score + epsilon)

    candidate_pcs = np.asarray(sorted(candidate_pcs_set))
    candidate_final_score = final_score[candidate_pcs]
    ranked_indices = np.argsort(candidate_final_score)[::-1][:n_pcs]
    selected_pcs = candidate_pcs[ranked_indices]

    info = {
        'final_score': final_score,
        'target_score': target_score,
        'noise_score': noise_score,
        'primary_target_pcs_lists': primary_target_pcs_lists,
        'target_pcs_lists': target_pcs_lists,
        'noise_pcs_lists': noise_pcs_lists,
    }
    return selected_pcs, info
