# 🔥 Modelo de Predição Industrial com Imagens Térmicas

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Meta DINOv2](https://img.shields.io/badge/DINOv2-ViT--B%2F14-green.svg)](https://github.com/facebookresearch/dinov2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Pipeline de visão computacional e aprendizado profundo voltado à **identificação e classificação automatizada de equipamentos elétricos de subestações de alta tensão a partir de imagens termográficas infravermelhas**.

O sistema utiliza o estado da arte em aprendizado auto-supervisionado com o backbone **DINOv2 ViT-B/14 (Meta AI)** para extração direta de embeddings visuais densos de 768 dimensões, sem necessidade de descritores manuais (dispensando SIFT/SURF) e sem modelos híbridos redundantes, garantindo robustez com aceleração **CUDA** (e fallback automático para CPU).

---

## 📸 Amostras dos Equipamentos Analisados

O dataset é composto por inspeções termográficas reais de 5 classes de ativos elétricos críticos:

![Amostras de Equipamentos Térmicos](docs/figures/sample_images.png)

| Equipamento | Função no Sistema Elétrico de Potência |
|---|---|
| **Circuit Breakers** (Disjuntores) | Manobra e interrupção de correntes de carga e curto-circuito em alta tensão. |
| **Disconnectors** (Seccionadoras) | Abertura visível de trechos de barramento e linhas para segurança de manutenção. |
| **Power Transformers** (Transformadores) | Conversão dos níveis de tensão entre transmissão e distribuição primária. |
| **Surge Arresters** (Para-raios) | Proteção contra sobretensões transitórias de origem atmosférica e manobras. |
| **Wave Traps** (Bobinas de Bloqueio) | Filtragem e injeção de sinais de telecomunicação na rede de transmissão. |

---

## 📊 Análise Exploratória e Distribuição de Classes

O dataset totaliza **893 imagens térmicas válidas** (com tratamento resiliente que identificou e ignorou automaticamente 2 imagens corrompidas no pipeline).

![Distribuição do Dataset](docs/figures/class_distribution.png)

A distribuição por classe no dataset:
- **Circuit Breakers:** 203 imagens (22.7%)
- **Disconnectors:** 180 imagens (20.2%)
- **Power Transformers:** 176 imagens (19.7%)
- **Surge Arresters:** 181 imagens (20.3%)
- **Wave Traps:** 153 imagens (17.1%)

---

## 🧠 Espaço Latente DINOv2 (Projeção t-SNE 2D)

O backbone **DINOv2 ViT-B/14** mapeia cada imagem térmica para um vetor de características denso de **768 dimensões** através do *CLS Token*. 

A projeção dimensional via **t-SNE (2D)** evidencia a segregação natural dos equipamentos no espaço euclidiano:

![Projeção t-SNE DINOv2](docs/figures/tsne_dinov2.png)

---

## 🔍 Auditoria de Overfitting e Desempenho Real

> [!WARNING]
> **Nota sobre Vazamento de Dados e Métricas Reais:**
> 1. **Viés de Resubstituição (In-Sample):** Avaliações preliminares que reportam 100% de precisão ocorrem quando o modelo é avaliado sobre os mesmos dados em que foi ajustado (*in-sample*). No conjunto de teste independente cego (*Hold-Out Test* de 135 imagens), a **acurácia real é de 98.52%** (133 acertos e 2 erros reais).
> 2. **Correlação Temporal por Sequências (Burst Effect):** Uma inspeção profunda no dataset revelou que as fotos provêm de rajadas de vídeo da câmera FLIR (ex: `FLIR2296`, `FLIR2298`...). Amostras consecutivas de um mesmo vídeo compartilham plano de fundo e ângulo. Para evitar alucinações de generalização, os modelos foram submetidos a **Validação Cruzada Out-of-Fold estrita (5-Fold CV)** e teste cego.

![Diagnóstico de Overfitting](docs/figures/overfitting_analysis.png)

### 📈 Tabela Comparativa de Generalização

| Conjunto de Avaliação | Amostras | Acurácia | Erros Observados | Observação |
|---|:---:|:---:|:---:|---|
| **Treino (In-sample)** | 626 | 100.0% | 0 | Memorização dos embeddings do treino |
| **Validação (Hold-out)** | 132 | 100.0% | 0 | Amostras decorrelacionadas da validação |
| **Teste Cego (Hold-out)** | 135 | **98.52%** | **2** | **133 de 135 imagens classificadas corretamente** |
| **Out-of-Fold (CV 5-Fold Total)** | 893 | **99.44%** | **5** | **Previsão cega para cada amostra fora do fold** |

---

## 🏆 Desempenho Comparativo dos Classificadores (5-Fold CV)

Validação cruzada estratificada de 5 folds sobre os embeddings de 768-D:

![Comparativo de Classificadores](docs/figures/classifier_comparison.png)

| Modelo Classificador | Acurácia Média (CV) | Desvio Padrão (±) | Tempo Médio CV |
|---|:---:|:---:|:---:|
| 🥇 **Logistic Regression** (L2, C=1.0) | **99.44%** | **± 0.50%** | **0.53s** |
| 🥈 **SVM** (RBF Kernel, C=10.0) | **99.44%** | **± 0.50%** | **0.58s** |
| 🥉 **Random Forest** (300 árvores) | **97.87%** | **± 1.57%** | **3.95s** |

---

## 🎯 Matriz de Confusão Real (Out-of-Fold)

A matriz de confusão abaixo reflete as **predições reais fora do fold (Out-of-Fold)**, onde cada imagem foi avaliada por um modelo que **nunca viu aquela amostra no treinamento**:

![Matriz de Confusão Real](docs/figures/confusion_matrix.png)

### 🔬 Análise dos 5 Casos de Falha Identificados:
1. **Circuit Breakers → Disconnectors (1 erro):** Disjuntor em ângulo lateral com lâminas abertas assemelhando-se visualmente a uma seccionadora.
2. **Surge Arresters → Wave Traps (1 erro):** Para-raios com topo cilíndrico confundido com a estrutura helicoidal de bobina de bloqueio.
3. **Surge Arresters → Circuit Breakers (1 erro):** Haste vertical confundida com a coluna de pólo de um disjuntor.
4. **Wave Traps → Power Transformers (1 erro):** Bobina em primeiro plano com transformador ao fundo no mesmo enquadramento térmico.
5. **Wave Traps → Surge Arresters (1 erro):** Bobina de bloqueio em plano afastado.

---

## 📋 Métricas Reais por Equipamento (Sem Overfitting)

![Métricas por Classe](docs/figures/metrics_per_class.png)

| Equipamento | Precisão (Precision) | Cobertura (Recall) | F1-Score | Amostras Reais |
|---|:---:|:---:|:---:|:---:|
| **Circuit Breakers** | **99.51%** | **99.51%** | **0.9951** | 203 |
| **Disconnectors** | **99.45%** | **100.0%** | **0.9972** | 180 |
| **Power Transformers** | **99.44%** | **100.0%** | **0.9972** | 176 |
| **Surge Arresters** | **99.44%** | **98.90%** | **0.9917** | 181 |
| **Wave Traps** | **99.34%** | **98.69%** | **0.9902** | 153 |
| **Média Ponderada (Weighted Avg)** | **99.44%** | **99.44%** | **0.9944** | **893** |

---

## 🏗️ Arquitetura do Repositório

Seguindo rigorosamente os princípios de **Clean Code** e **SOLID**:

```
modelo-de-predicao-industrial-com-imagens-termicas/
│
├── README.md                           # Documentação detalhada e auditoria de métricas
├── requirements.txt                    # Dependências do projeto
├── .gitignore                          # Ignora datasets brutos e binários pesados
├── run_pipeline.py                     # Script para execução ponta-a-ponta do pipeline
│
├── docs/
│   └── figures/                        # Gráficos em alta resolução (300 DPI)
│       ├── class_distribution.png      # Distribuição do dataset
│       ├── sample_images.png           # Amostras das 5 classes
│       ├── tsne_dinov2.png             # Projeção t-SNE 2D do espaço latente
│       ├── classifier_comparison.png   # Boxplot comparativo dos 3 modelos (5-Fold)
│       ├── confusion_matrix.png        # Matriz de confusão real Out-of-Fold
│       ├── metrics_per_class.png       # Precisão, Recall e F1 por classe
│       └── overfitting_analysis.png    # Gráfico de análise de gap de overfitting
│
├── notebooks/
│   ├── 01_preprocessing.ipynb          # Notebook didático: EDA e geração de splits
│   └── 02_training.ipynb               # Notebook didático: DINOv2 + Classificação
│
├── src/
│   ├── __init__.py                     # Interface do pacote
│   ├── config.py                       # Dataclass imutável DINOv2Config
│   ├── transforms.py                   # Pipeline canônico torchvision (Resize bicúbico + Norm)
│   ├── dataset.py                    # Dataset PyTorch com safe loading de arquivos
│   └── feature_extractor.py            # Extrator de embeddings DINOv2 ViT-B/14 (CUDA/CPU)
│
├── data/
│   ├── raw/                            # Imagens térmicas originais organizadas por classe
│   └── processed/                      # Splits train.pt, val.pt, test.pt gerados
│
└── outputs/
    ├── features/                       # Embeddings 768-D extraídos (features_dinov2.npy)
    └── models/
        ├── metrics_summary.json        # Resumo quantitativo das métricas reais em JSON
        └── classifier_dinov2.pkl       # Melhor modelo serializado pronto para produção
```

---

## 🚀 Como Executar

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone https://github.com/gabrielsena87/modelo-de-predicao-industrial-com-imagens-termicas.git
cd modelo-de-predicao-industrial-com-imagens-termicas
pip install -r requirements.txt
```

### 2. Execução Automatizada (Pipeline Completo)
Para processar os dados, extrair features, rodar o t-SNE, treinar os classificadores e exportar todos os gráficos:
```bash
python run_pipeline.py
```

### 3. Execução Interativa via Jupyter Notebooks
Se preferir executar passo a passo de forma didática:
1. Abra e execute [`notebooks/01_preprocessing.ipynb`](notebooks/01_preprocessing.ipynb) para analisar os dados e gerar os splits.
2. Abra e execute [`notebooks/02_training.ipynb`](notebooks/02_training.ipynb) para extrair os embeddings DINOv2 e comparar os classificadores.

---

## 🛠️ Stack Tecnológica

- **Visão Computacional & Deep Learning:** PyTorch, Torchvision, Meta DINOv2 ViT-B/14
- **Machine Learning & Validação:** Scikit-Learn (Logistic Regression, SVM, Random Forest, Stratified K-Fold, t-SNE)
- **Visualização e Métricas:** Matplotlib, Seaborn
- **Manipulação de Imagens & Dados:** Pillow, NumPy

---

## 📄 Licença

Este projeto é desenvolvido para fins educacionais, acadêmicos e de pesquisa em manutenção preditiva industrial. Distribuído sob a licença MIT.
