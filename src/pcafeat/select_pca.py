import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pcafeat.pca_feature_select import pca_extract, con_extract
import os


def select_pca_features(df_X_train, target, method_pick_pca='fdr_bh',
                        method_pick_con='fdr_bh', fig_plot=False,
                        fig_dir=None, bar_color='blue'):
    if not isinstance(df_X_train, pd.DataFrame):
        raise TypeError("df_X_train must be a pandas.DataFrame")

    if len(df_X_train) != len(target):
        raise ValueError("df_X_train and target must have the same number of samples")

    if fig_plot and fig_dir is None:
        raise ValueError("fig_dir must be provided when fig_plot=True")

    if fig_plot and not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    pca = PCA().fit(df_X_train)
    coeff_train = pca.components_.T
    score = pca.transform(df_X_train)

    n_components = score.shape[1]
    pca_id = [f'pca_score{i}' for i in range(1, n_components + 1)]
    df_score = pd.DataFrame(score, columns=pca_id)

    select_pc, statistics = pca_extract(
        df_score,
        target,
        method_pick_pca,
        fig_plot=fig_plot,
        fig_dir=fig_dir,
        bar_color=bar_color
    )

    cons, cons_pc = con_extract(coeff_train, select_pc, method_pick_con)

    return cons, cons_pc
