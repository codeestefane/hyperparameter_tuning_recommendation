import os

from sklearn import metrics

import numpy as np

# from default_experiment import seeds, algorithms

# %% 
# cria arquivos com os resultados obtidos por cada combinação (setup) do experimento
def write_csv_result(tipo_experimento, file, bool_missing_value, abordagem_missing_value, algoritmo, threshold, test_idx, resultados, f1_score, acc_balanceada, auc_score, seed, counter, seeds):
    # se não tinha valores faltantes no dataset
    if not bool_missing_value:
        # não foi utilizada nenhuma abordagem para tratar valores faltantes
        abordagem_missing_value = None

    os.makedirs('../resultados/' + tipo_experimento + '/resultados_combinacoes/', exist_ok = True)

    with open('../resultados/' + tipo_experimento + '/resultados_combinacoes/' + file + '_' + str(abordagem_missing_value) + '_' + algoritmo + '_' + str(threshold).replace(".", "_") + '.csv', 'a') as f:
        # se estiver na primeira seed e na primeira iteração da validação cruzada
        if seed == seeds[0] and counter == 0:
            # escreve o cabeçalho do arquivo
            f.write("seed;iteracao_validacao_cruzada;indices;previsoes;f1_score;acuracia_balanceada;auc\n")

        # resultados obtidos por seed e iteração
        f.write(";".join([str(seed), str(counter), str(test_idx), str(resultados), str(f1_score), str(acc_balanceada), str(auc_score)]) + "\n")

# %%
def identify_analized_features(pipeline, bool_missing_value, approach, algorithm, algorithms):
    # se a abordagem de tratamento de valores faltantes for a primeira e o pipeline estiver no primeiro algoritmo 
    # OBS: condição criada só para analisar as features com valores faltantes e não correlacionadas uma vez por seed e iteração da validação cruzada (pré-processamento não tem comportamento estocástico)
    if approach == "remove_missing_values" and algorithm == list(algorithms.keys()).pop(0):
        features_missing_values = []

        # identifica features que permaneceram após a remoção das features correlacionadas
        not_corr_features = list(pipeline.named_steps['preprocessing'].named_steps['correlated_features'].data.columns[0:-1])

        # se tiver valores faltantes no dataset
        if bool_missing_value:
            # identifica as features com valores faltantes
            features_missing_values = list(pipeline.named_steps['preprocessing'].named_steps['missing_values'].features_missing_values)

        # senão
        else:
            # atribui None
            features_missing_values = None

        return features_missing_values, not_corr_features

    return None, None
    
# %%
# cria arquivos com o resumo das features que possuem valores faltantes; também salva features não correlacionadas para identificar as features consideradas no treinamento de cada modelo 
def write_csv_analyzed_features(tipo_experimento, file, threshold, features_nao_correlacionadas, features_missing_values, seed, counter, seeds):
    # se tiver features não correlacionadas para analisar
    if features_nao_correlacionadas != None:
        os.makedirs('../resultados/'+ tipo_experimento + '/features/', exist_ok = True)

        with open('../resultados/' + tipo_experimento + '/features/' + file + str(threshold).replace(".", "_") + '.csv', 'a') as f:
             # se estiver na primeira seed e na primeira iteração da validação cruzada
            if seed == seeds[0] and counter == 0:
                # escreve cabeçalho do arquivo
                f.write("seed;iteracao_validacao_cruzada;features_nao_correlacionadas;features_missing_values\n")

            # escreve resultado obtido
            f.write(";".join([str(seed), str(counter), str(features_nao_correlacionadas), str(features_missing_values)]) + "\n")

# %%
# métricas calculadas para análise de resultados
def calculate_metrics(y, predict, positive_proba):
    f1 = metrics.f1_score(y, predict)

    balanced_acc = metrics.balanced_accuracy_score(y, predict)
    
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), positive_proba)
    auc_score = metrics.auc(fpr, tpr)

    return f1, balanced_acc, auc_score