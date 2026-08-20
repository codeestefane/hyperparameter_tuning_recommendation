# Experimentos de classificação com features ELA

Este projeto avalia classificadores de machine learning aplicados a meta-datasets com features de *Exploratory Landscape Analysis* (ELA). O objetivo é investigar a classificação das instâncias nas classes `Defaults` e `Tuning` usando diferentes conjuntos de features, estratégias de tratamento de valores ausentes e limiares para remoção de features correlacionadas.

## Por que este projeto é útil

O repositório reúne uma implementação reproduzível para comparar configurações de classificação sob as mesmas condições experimentais:

- pré-processamento com remoção de features constantes e altamente correlacionadas;
- tratamento de valores ausentes por remoção, imputação pela média, imputação KNN ou criação de indicadores de erro em grupos de features ELA;
- comparação entre Naive Bayes, árvore de decisão, KNN, Random Forest, SVM linear, SVM com kernel RBF, regressão logística e XGBoost;
- validação cruzada estratificada com 10 partições e 10 sementes aleatórias;
- avaliação por F1-score, acurácia balanceada e AUC-ROC;
- tuning de hiperparâmetros com otimização bayesiana no fluxo específico de tuning;
- registro dos resultados por dataset, algoritmo, estratégia de valores ausentes, limiar de correlação, semente e iteração.

## Estrutura do projeto

```text
.
├── datasets/                 # Meta-datasets usados pelos scripts
├── scripts/
│   ├── default_experiment.py # Execução com parâmetros padrão
│   └── tuning_experiment.py  # Execução com tuning bayesiano
├── resultados/               # CSVs de resultados e features analisadas
├── results.ipynb             # Exploração e análise dos resultados
└── requirements.txt          # Dependências Python fixadas
```

Os scripts esperam ser executados a partir da raiz do repositório, pois usam caminhos relativos como `./datasets/<arquivo>.csv` e `./resultados/<experimento>/`.

## Requisitos

- Python compatível com as versões das dependências fixadas em `requirements.txt`;
- ambiente virtual recomendado;
- espaço em disco e tempo de execução suficientes para os experimentos completos. O tuning combina validação externa, validação interna e otimização de hiperparâmetros, portanto pode ser demorado.

## Instalação

No Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Como executar

### Experimento padrão

Com o ambiente virtual ativado e estando na raiz do projeto:

```bash
python scripts/default_experiment.py
```

O script percorre os datasets configurados em `scripts/default_experiment.py`, os algoritmos, os limiares `0.8`, `0.85`, `0.9` e `0.95`, as estratégias aplicáveis de valores ausentes e as 10 sementes. Os arquivos são criados ou atualizados em `resultados/default/`, incluindo CSVs de métricas e de features analisadas.

### Experimento com tuning

```bash
python scripts/tuning_experiment.py
```

O script usa `scikit-optimize` para otimizar os hiperparâmetros de cada algoritmo e grava os resultados em `resultados/tuning/`.

Os arquivos detalhados ficam em `resultados/tuning/resultados_combinacoes/` e `resultados/tuning/features/`. O tuning usa os mesmos tres datasets listados na secao de dados de entrada.

### Análise em notebook

Para executar as versões interativas e visualizar as análises:

```bash
jupyter lab
```

Abra `results.ipynb` para explorar os CSVs gerados. Os experimentos executáveis nesta cópia do projeto estão nos scripts Python; mantenha o diretório de trabalho do notebook na raiz para preservar os caminhos relativos.

## Dados de entrada

Os scripts leem CSVs separados por vírgula em `datasets/`. Cada dataset deve conter uma coluna identificadora na primeira posição, as features entre a primeira e a última coluna e a variável-alvo na última coluna. Os valores esperados para a variável-alvo são `Defaults` e `Tuning`; eles são convertidos internamente para `0` e `1`.

Os arquivos atualmente usados pelo experimento padrão são:

- `ela_features_flacco.csv`;
- `classif_svm_169d_95_average.csv`;
- `classif_svm_ela_features_flacco.csv`.

## Resultados

Os CSVs de métricas registram, por configuração, a semente, a iteração da validação cruzada, os índices de teste, as previsões, o F1-score, a acurácia balanceada e a AUC. Os arquivos em `resultados/*/features/` registram as features não correlacionadas e, quando aplicável, as features com valores ausentes identificadas durante o pré-processamento.

Os notebooks de resultados podem ser usados para consolidar e visualizar esses arquivos sem alterar os dados de entrada.

## Desenvolvimento

Para reproduzir ou adaptar um experimento, ajuste no script correspondente as listas de datasets, algoritmos, sementes, limiares e espaços de busca. Execute sempre a partir da raiz e confirme que os arquivos de entrada estão no caminho esperado. O projeto não inclui atualmente uma suíte de testes automatizados; a validação principal é feita pela execução dos notebooks e pela inspeção dos CSVs produzidos.
