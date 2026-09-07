# Experimento ELA + metadatasets para classificação

Este projeto avalia diferentes combinações de metadatasets, estratégias de tratamento de valores ausentes, limiares de correlação e algoritmos de aprendizado de máquina para classificar instâncias de tuning vs. defaults em problemas de otimização. A análise combina recursos de ELA (Evolutionary Landscape Analysis), metadados de desempenho e métricas estatísticas para comparar configurações em vários cenários.

## O que o projeto faz

O repositório organiza um pipeline experimental para:

- carregar metadatasets de diferentes fontes;
- remover ou tratar colunas com valores ausentes;
- eliminar atributos constantes e altamente correlacionados;
- normalizar features antes do treinamento;
- avaliar modelos com validação estratificada;
- comparar desempenho por métrica e por semente aleatória;
- gerar resumos e gráficos de análise estatística.

Os arquivos principais estão em `scripts/` e a execução principal é feita por meio de scripts separados para experimentos, agregação de métricas e análise interpretável.

## Por que este projeto é útil

O projeto foi estruturado para suportar pesquisas em benchmarking de algoritmos e análise de metadados de desempenho. Ele é útil porque:

- permite comparar diferentes abordagens de imputação e remoção de atributos;
- considera vários limiares de correlação para reduzir redundância entre features;
- executa experimentos com diversos algoritmos, incluindo SVM, k-NN, Random Forest, regressão logística, Naive Bayes, Decision Tree e XGBoost;
- gera arquivos CSV com métricas de F1, balanced accuracy e AUC por iteração e por semente;
- produz gráficos e análises estatísticas com Friedman + Nemenyi para comparar configurações;
- inclui uma etapa de SHAP para interpretar o impacto das features em um modelo.

Em resumo, a estrutura ajuda a reproduzir experimentos e a documentar quais combinações de preprocessing e modelo performam melhor.

## Estrutura do repositório

```text
.
├── datasets/
│   ├── classif_svm_169d_95_average.csv
│   ├── classif_svm_ela_features_flacco.csv
│   ├── ela_features_flacco.csv
│   └── ...
├── resultados/
│   └── default/
│       ├── features/
│       ├── plots/
│       ├── resultados_combinacoes/
│       ├── resumo_resultado_default.csv
│       └── shap_values.csv
├── scripts/
│   ├── 01_default_experiment.py
│   ├── 02_generate_results_resume.py
│   ├── 03_friedman_nemenyi_tests.py
│   ├── 04_generate_aggregate_metrics.py
│   ├── 05_shap_analysis.py
│   └── modules/
│       ├── pipeline.py
│       ├── preprocessing.py
│       └── report_results.py
├── requirements.txt
└── README.md
```

## Pré-requisitos

Este projeto depende principalmente de:

- Python 3.10+
- pandas, numpy, scipy
- scikit-learn
- xgboost
- shap
- seaborn, matplotlib
- scikit-optimize
- scikit-posthocs
- tqdm

A lista completa está em [requirements.txt](requirements.txt).

## Como começar

### 1) Clone o repositório

```bash
git clone <url-do-repositorio>
cd experimento_ela_features
```

### 2) Crie um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
# no Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
```

### 3) Instale as dependências

```bash
pip install -r requirements.txt
```

### 4) Execute os experimentos

Os scripts em `scripts/` usam caminhos relativos a partir da pasta `scripts` do projeto. Por isso, a execução recomendada é:

```bash
cd scripts
python 01_default_experiment.py
```

Esse script executa o experimento principal com:

- metadatasets: `ela_features_flacco`, `classif_svm_169d_95_average`, `classif_svm_ela_features_flacco`;
- limiares de correlação: `0.8`, `0.85`, `0.9`, `0.95`;
- abordagens de missing values: `remove_missing_values`, `imputer_mean`, `knn_imputer`, `set_ela_error`;
- algoritmos: `NB`, `DT`, `KNN`, `RF`, `SVM_RBF`, `SVM_LIN`, `LogisticRegression`, `XGBoost`;
- sementes: `[3, 5, 7, 13, 27, 35, 42, 66, 72, 111]`.

Os resultados são salvos em `resultados/default/resultados_combinacoes/` e também em `resultados/default/features/`.

### 5) Gere o resumo dos resultados

```bash
cd scripts
python 02_generate_results_resume.py
```

Esse passo consolida as combinações em um arquivo como:

- `resultados/default/resumo_resultado_default.csv`

### 6) Gere testes estatísticos e gráficos

```bash
cd scripts
python 03_friedman_nemenyi_tests.py
```

O script aplica testes de Friedman e Nemenyi e salva diagramas críticos em `resultados/default/plots/`.

### 7) Gere métricas agregadas

```bash
cd scripts
python 04_generate_aggregate_metrics.py
```

Esse script produz tabelas agregadas com métricas consolidadas para comparação entre metadatasets.

### 8) Execute a análise SHAP

```bash
cd scripts
python 05_shap_analysis.py
```

O arquivo gera valores SHAP e salva a visualização no diretório `resultados/default/plots/` e `resultados/default/shap_values.csv`.

## Exemplos de uso

### Executar o experimento principal

```bash
cd scripts
python 01_default_experiment.py
```

### Gerar resumo depois do experimento

```bash
cd scripts
python 02_generate_results_resume.py
```

### Visualizar resultados já gerados

```python
import pandas as pd

resumo = pd.read_csv('../resultados/default/resumo_resultado_default.csv', sep=';')
print(resumo.head())
```

## Scripts relevantes

- [scripts/01_default_experiment.py](scripts/01_default_experiment.py): executa o pipeline completo de avaliação.
- [scripts/02_generate_results_resume.py](scripts/02_generate_results_resume.py): consolida métricas por combinação.
- [scripts/03_friedman_nemenyi_tests.py](scripts/03_friedman_nemenyi_tests.py): testes estatísticos e diagramas de comparação.
- [scripts/04_generate_aggregate_metrics.py](scripts/04_generate_aggregate_metrics.py): tabela agregada de métricas.
- [scripts/05_shap_analysis.py](scripts/05_shap_analysis.py): análise de importância de features com SHAP.

Os módulos reutilizáveis estão em:

- [scripts/modules/pipeline.py](scripts/modules/pipeline.py)
- [scripts/modules/preprocessing.py](scripts/modules/preprocessing.py)
- [scripts/modules/report_results.py](scripts/modules/report_results.py)

## Observações importantes

- O projeto foi organizado para análise experimental e comparação de configurações.
- Muitos scripts dependem de caminhos relativos e funcionam corretamente quando executados a partir da pasta `scripts/`.
- Os arquivos gerados ficam em `resultados/default`, que é o diretório padrão para análise do experimento.

## Contribuição

Para contribuir com melhorias, mantenha o mesmo padrão de organização do projeto e valide os scripts antes de enviar alterações. Em geral, mudanças em preprocessing, métricas ou análise estatística devem ser acompanhadas de execução local dos scripts relevantes para garantir a reprodução dos resultados.
