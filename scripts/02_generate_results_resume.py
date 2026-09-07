# %%
import os
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import ast

from tqdm import tqdm 

# %%
resultados_list = os.listdir("../resultados/default/resultados_combinacoes/")

# %%
files = ['ela_features_flacco', 'classif_svm_169d_95_average', 'classif_svm_ela_features_flacco']

# %%
resultados = {"arquivo": [], "media_f1_score": [], "mediana_f1_score": [], "desvio_padrao_f1_score": [], "min_f1_score": [], "max_f1_score": [], "varianca_f1_score": [], "media_acc_balanceada": [], "mediana_acc_balanceada": [], "desvio_padrao_acc_balanceada": [], "min_acc_balanceada": [], "max_acc_balanceada": [], "varianca_acc_balanceada": [], "media_auc": [], "mediana_auc": [], "desvio_padrao_auc": [], "min_auc": [], "max_auc": [], "varianca_auc": [], "instancias_mais_erradas": []}

# %%
# analisa cada um dos arquivos e salva os resultados em um arquivo de resumo
for r in tqdm(resultados_list):
    dados = pd.read_csv("../resultados/default/resultados_combinacoes/" + r, sep = ';')
    resultados["arquivo"].append(r)
   
    seeds = np.unique(dados['seed'].values)

    medias_f1_score_seed = []
    medias_auc_seed = []
    medias_acc_balanceada_seed = []

    for s in seeds:
        medias_f1_score_seed.append(np.mean([dados['f1_score'][i] for i, j in enumerate(dados['seed']) if j == s]))
        medias_auc_seed.append(np.mean([dados['auc'][i] for i, j in enumerate(dados['seed']) if j == s]))
        medias_acc_balanceada_seed.append(np.mean([dados['acuracia_balanceada'][i] for i, j in enumerate(dados['seed']) if j == s]))

    resultados["desvio_padrao_f1_score"].append(np.std(medias_f1_score_seed))
    resultados["desvio_padrao_auc"].append( np.std(medias_auc_seed))
    resultados["desvio_padrao_acc_balanceada"].append(np.std(medias_acc_balanceada_seed))
    
    resultados["varianca_f1_score"].append(np.var(medias_f1_score_seed))
    resultados["varianca_auc"].append(np.var(medias_auc_seed))
    resultados["varianca_acc_balanceada"].append(np.var(medias_acc_balanceada_seed))

    resultados["min_f1_score"].append(min(medias_f1_score_seed))
    resultados["min_auc"].append(min(medias_auc_seed))
    resultados["min_acc_balanceada"].append(min(medias_acc_balanceada_seed))

    resultados["max_f1_score"].append(max(medias_f1_score_seed))
    resultados["max_auc"].append(max(medias_auc_seed))
    resultados["max_acc_balanceada"].append(max(medias_acc_balanceada_seed))

    resultados["media_f1_score"].append(np.mean(medias_f1_score_seed))
    resultados["media_auc"].append(np.mean(medias_auc_seed))
    resultados["media_acc_balanceada"].append(np.mean(medias_acc_balanceada_seed))

    resultados["mediana_f1_score"].append(np.median(medias_f1_score_seed))
    resultados["mediana_auc"].append(np.median(medias_auc_seed))
    resultados["mediana_acc_balanceada"].append(np.median(medias_acc_balanceada_seed))

    for f in files:
        if f in r:
            resultado = pd.read_csv("../datasets/" + f + ".csv")

    instancias_erradas = []

    for i in range(len(dados)):
        previsoes = ast.literal_eval(dados["previsoes"][i])
        indices_instancias = ast.literal_eval(str(dados["indices"][i]))
        gabarito = list(resultado["Class"][indices_instancias])

        comparacao = []

        for g, j in zip(gabarito, previsoes):
            comparacao.append(g == j)

        instancias_erradas.append([indices_instancias[k] for k, j in enumerate(comparacao) if not j])

    frequencia_maxima = max(Counter(list(itertools.chain.from_iterable(instancias_erradas))).values())
    frequencia_instancias_erradas = list(Counter(list(itertools.chain.from_iterable(instancias_erradas))).items())
    instancias_erradas = [i[0] for i in frequencia_instancias_erradas if i[1] >= frequencia_maxima]
    arquivos_mais_classificados_erroneamente = list(resultado[resultado.columns[0]][instancias_erradas].values)

    resultados["instancias_mais_erradas"].append(arquivos_mais_classificados_erroneamente)

resultados = pd.DataFrame(resultados)

resultados.to_csv('../resultados/default/resumo_resultado_default.csv', index = False, encoding = 'utf-8', sep = ";")