# 🔥 Modelo de Predição Industrial com Imagens Térmicas

Pipeline de classificação de equipamentos elétricos em imagens térmicas infravermelhas utilizando features extraídas pelo **DINOv2 ViT-B/14** (Meta AI) com aceleração **CUDA**.

## 📋 Classes de Equipamentos

| Classe | Descrição |
|---|---|
| Circuit Breakers | Disjuntores |
| Disconnectors | Seccionadoras |
| Power Transformers | Transformadores de potência |
| Surge Arresters | Para-raios |
| Wave Traps | Bobinas de bloqueio |

## 🏗️ Estrutura do Projeto

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/                          # Dataset original (5 classes)
├── notebooks/
│   ├── 01_preprocessing.ipynb        # EDA + pré-processamento
│   └── 02_training.ipynb             # Extração DINOv2 + treino
├── src/
│   ├── __init__.py
│   ├── config.py                     # Dataclass de configuração
│   ├── transforms.py                 # Pipeline de transformações
│   └── dataset.py                    # Dataset customizado PyTorch
│   └── feature_extractor.py          # Extrator DINOv2 + CUDA
└── outputs/
    ├── features/                     # Features extraídas (.npy)
    └── models/                       # Modelos treinados (.pkl)
```

## 🚀 Quickstart

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Organizar o dataset
Coloque as imagens térmicas em `data/raw/`, uma subpasta por classe:
```
data/raw/
├── Circuit Breakers/
├── Disconnectors/
├── Power Transformers/
├── Surge Arresters/
└── Wave Traps/
```

### 3. Executar os notebooks
1. **`notebooks/01_preprocessing.ipynb`** — Análise exploratória e validação do dataset.
2. **`notebooks/02_training.ipynb`** — Extração de features DINOv2 e treinamento do classificador.

## ⚙️ Stack Tecnológica

- **Backbone**: DINOv2 ViT-B/14 (768-dim embeddings)
- **Aceleração**: CUDA (GPU NVIDIA)
- **Classificador**: Scikit-learn (Logistic Regression / SVM / Random Forest)
- **Framework**: PyTorch + torchvision

## 📊 Resultados

Resultados detalhados disponíveis nos notebooks após execução.

## 📝 Licença

Projeto acadêmico — uso educacional.
