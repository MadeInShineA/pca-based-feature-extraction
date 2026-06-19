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


def pca_extract(df_score, target, method_pick_pca,  fig_plot, fig_dir, multipletests_alpha=0.05, bar_color='blue'):
    """
    df_score: PCAスコアのみを含むDataFrame（NaNは既に除外されている想定）
    target: ターゲット変数のSeries（df_score[target]に相当、NaNは既に除外されている想定）
    bar_color: プロットのバーの色（デフォルト: 'blue'）
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

    h = smt.multipletests(p, alpha=multipletests_alpha, method=method_pick_pca)[0]
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
