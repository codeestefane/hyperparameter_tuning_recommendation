import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

resultados = pd.read_csv("../resultados/default/resumo_resultado_default.csv", sep = ';')

metadatasets = {'ela_features_flacco': [], 'classif_svm_169d_95_average': [], 'classif_svm_ela_features_flacco': []}

for r in range(len(resultados)):
    for md in metadatasets:
        if md in resultados.loc[r]["arquivo"] and ("None" in resultados.loc[r]["arquivo"] or "remove_missing_values" in resultados.loc[r]["arquivo"]):
            if md == 'ela_features_flacco' and 'classif_svm' in resultados.loc[r]["arquivo"]:
                pass
            else:
                # substituir "media_acc_balanceada" por "media_f1_score" ou "media_auc" dependendo da métrica que você deseja analisar
                metadatasets[md].append((resultados.loc[r]["arquivo"], resultados.loc[r]["media_acc_balanceada"]))

dataframes_scores_ordenados = {"classif_svm_169d_95_average": None, "ela_features_flacco": None, "classif_svm_ela_features_flacco": None}

for d in dataframes_scores_ordenados:
    scores_ordenados = {'0_8.': [], '0_85': [], '0_9.': [], '0_95': []}
    for alg in ["NB", "DT", "KNN", "RF", "SVM_RBF", "SVM_LIN", "LogisticRegression", "XGB"]:
        for i in metadatasets[d]:
            if alg in i[0]:
                for corr in scores_ordenados:
                    if corr in i[0]:
                        scores_ordenados[corr].append(i[1])

    dataframes_scores_ordenados[d] = pd.DataFrame(scores_ordenados, index = ["NB", "DT", "KNN", "RF", "SVM_RBF", "SVM_LIN", "RL", "XGB"])
    dataframes_scores_ordenados[d].columns = ['0.8', '0.85', '0.9', '0.95']

    dfs = [dataframes_scores_ordenados[d] for d in dataframes_scores_ordenados]

titles = ["Baseline", "Ela Meta-Features (Flacco)", "Combined"]

fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)

custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_heatmap", ["#fcb6a8", "#ffded8", "#fdfdfd", "#a9c7ff"])

all_values = np.concatenate([df.values.flatten() for df in dfs])
vmin, vmax = all_values.min(), all_values.max()

for i, ax in enumerate(axes):
    sns.heatmap(
    dfs[i], 
    annot=True, 
    fmt=".3f", 
    cmap = custom_cmap,
    vmin=vmin, vmax=vmax,
    cbar=False, 
    linewidths=1.5, 
    linecolor='white',
    annot_kws={"size": 11, "color": "black"},
    ax=ax
    )

    header = Rectangle((0, 1), 1, 0.05, transform=ax.transAxes, 
                    facecolor='#E5E5E5', edgecolor='black', linewidth=1, clip_on=False)
    ax.add_patch(header)

    ax.text(0.5, 1.025, titles[i], transform=ax.transAxes, 
            ha='center', va='center', fontsize=12, fontweight='normal')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    if i == 0:
        ax.tick_params(axis='both', which='both', length=4, width=1, colors='black', labelsize=11)
        ax.tick_params(axis='y', labelrotation=0)
    else:
        ax.tick_params(left=False, labelleft=False)

plt.subplots_adjust(wspace=0.025, right=0.90, top=0.90)

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7]) 
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)

# substituir 'BAC' por 'F\u2081-score' ou 'AUC' dependendo da métrica que você deseja analisar
cbar.set_label('BAC', fontsize=11, labelpad=10)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

# substituir "acc_balanced" por "f1_score" ou "auc" dependendo da métrica que você deseja analisar
plt.savefig('../resultados/default/plots/acc_balanced_aggregated_metric_table.png', dpi=300, bbox_inches='tight')
plt.show()