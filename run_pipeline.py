"""
Pipeline completo de execução, avaliação e geração de relatórios gráficos.
Executa todo o fluxo:
1. Análise Exploratória e Divisão de Dados (Train/Val/Test)
2. Extração de Embeddings DINOv2 ViT-B/14 (768 dimensões)
3. Projeção t-SNE 2D do espaço latente
4. Validação Cruzada Estratificada (5-Fold) comparativa (LogReg, SVM, Random Forest)
5. Treinamento final e Matriz de Confusão / Relatório por Classe
6. Exportação das figuras em alta resolução para docs/figures/
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.config import DINOv2Config
from src.dataset import ThermalImageDataset, safe_collate_fn
from src.feature_extractor import DINOv2FeatureExtractor
from src.transforms import build_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def setup_directories() -> tuple[Path, Path, Path, Path]:
    figures_dir = Path("docs/figures")
    processed_dir = Path("data/processed")
    features_dir = Path("outputs/features")
    models_dir = Path("outputs/models")

    for d in [figures_dir, processed_dir, features_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return figures_dir, processed_dir, features_dir, models_dir


def step1_eda_and_splits(config: DINOv2Config, figures_dir: Path, processed_dir: Path):
    logger.info("=== PASSO 1: Análise Exploratória e Divisão dos Dados ===")
    
    # Dataset bruto para inspecionar
    raw_dataset = ThermalImageDataset(config, transform=None)
    logger.info(f"Total de imagens carregadas: {len(raw_dataset)} em {len(raw_dataset.classes)} classes.")

    labels = [label for _, label in raw_dataset.samples]
    class_counts = Counter(labels)
    class_names = raw_dataset.classes

    # Plot 1: Distribuição de Classes
    plt.figure(figsize=(9, 5), dpi=300)
    colors = sns.color_palette("mako", len(class_names))
    x_indices = np.arange(len(class_names))
    counts = [class_counts[i] for i in x_indices]
    
    bars = plt.bar(x_indices, counts, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
    plt.xticks(x_indices, class_names, rotation=15, ha="right", fontsize=11, fontweight="bold")
    plt.ylabel("Número de Imagens", fontsize=12, fontweight="bold")
    plt.title("Distribuição do Dataset de Imagens Térmicas por Equipamento", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(raw_dataset)) * 100
        plt.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.ylim(0, max(counts) + 35)
    plt.tight_layout()
    dist_path = figures_dir / "class_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    logger.info(f"Salvo gráfico de distribuição em: {dist_path}")

    # Plot 2: Amostras visuais
    fig, axes = plt.subplots(1, len(class_names), figsize=(18, 4), dpi=300)
    for cls_idx, cls_name in enumerate(class_names):
        found = False
        for img_path, label in raw_dataset.samples:
            if label == cls_idx:
                try:
                    img = Image.open(img_path).convert("RGB")
                    axes[cls_idx].imshow(img)
                    axes[cls_idx].set_title(cls_name, fontsize=11, fontweight="bold")
                    axes[cls_idx].axis("off")
                    found = True
                    break
                except Exception:
                    continue
        if not found:
            axes[cls_idx].axis("off")

    plt.suptitle("Amostras Representativas das Classes de Equipamentos Térmicos", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    samples_path = figures_dir / "sample_images.png"
    plt.savefig(samples_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Salvo mosaico de amostras em: {samples_path}")

    # Geração dos splits com transformações
    transform = build_transforms(config)
    transformed_dataset = ThermalImageDataset(config, transform=transform)

    torch.manual_seed(42)
    total = len(transformed_dataset)
    train_len = int(0.70 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len

    train_ds, val_ds, test_ds = random_split(transformed_dataset, [train_len, val_len, test_len])
    logger.info(f"Splits gerados: Treino={len(train_ds)}, Validação={len(val_ds)}, Teste={len(test_ds)}")

    for split_ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        loader = DataLoader(split_ds, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=safe_collate_fn)
        all_t, all_l = [], []
        for batch in loader:
            if batch is None:
                continue
            imgs, lbls, _ = batch
            all_t.append(imgs)
            all_l.append(lbls)
        
        tensors = torch.cat(all_t)
        labels_t = torch.cat(all_l)
        torch.save({"images": tensors, "labels": labels_t, "classes": class_names}, processed_dir / f"{name}.pt")
        logger.info(f"Salvo split {name}.pt com {len(tensors)} amostras.")

    return transformed_dataset, class_names


def step2_extract_features(config: DINOv2Config, dataset: ThermalImageDataset, features_dir: Path):
    logger.info("=== PASSO 2: Extração de Features DINOv2 ViT-B/14 ===")
    extractor = DINOv2FeatureExtractor(config)

    t0 = time.time()
    features, labels, paths = extractor.extract_from_dataset(dataset)
    elapsed = time.time() - t0

    logger.info(f"Extração finalizada em {elapsed:.2f}s ({elapsed / len(features):.3f}s/imagem).")
    logger.info(f"Matriz de embeddings: shape={features.shape}, dtype={features.dtype}")

    np.save(features_dir / "features_dinov2.npy", features)
    np.save(features_dir / "labels.npy", labels)

    return features, labels, paths, elapsed


def step3_tsne_projection(features: np.ndarray, labels: np.ndarray, class_names: list[str], figures_dir: Path):
    logger.info("=== PASSO 3: Projeção Dimensional com t-SNE ===")
    tsne = TSNE(n_components=2, perplexity=30, n_iter_without_progress=300, random_state=42)
    embeddings_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8), dpi=300)
    palette = sns.color_palette("tab10", len(class_names))

    for idx, name in enumerate(class_names):
        mask = (labels == idx)
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=name,
            color=palette[idx],
            alpha=0.85,
            edgecolors="none",
            s=45,
        )

    plt.title("Projeção t-SNE 2D dos Embeddings DINOv2 (768-D)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Dimensão t-SNE 1", fontsize=11, fontweight="bold")
    plt.ylabel("Dimensão t-SNE 2", fontsize=11, fontweight="bold")
    plt.legend(title="Equipamentos", fontsize=10, title_fontsize=11, loc="best", frameon=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    tsne_path = figures_dir / "tsne_dinov2.png"
    plt.savefig(tsne_path, dpi=300)
    plt.close()
    logger.info(f"Salvo gráfico t-SNE em: {tsne_path}")


def step4_train_and_evaluate(features: np.ndarray, labels: np.ndarray, class_names: list[str], figures_dir: Path, models_dir: Path):
    logger.info("=== PASSO 4: Treinamento e Validação Cruzada Estratificada ===")
    
    classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=42))
        ]),
        "SVM (RBF Kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results: dict[str, np.ndarray] = {}

    for name, model in classifiers.items():
        t_start = time.time()
        scores = cross_val_score(model, features, labels, cv=cv, scoring="accuracy")
        t_dur = time.time() - t_start
        cv_results[name] = scores
        logger.info(f"Modelo: {name:<22} | Acurácia CV: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}% | Tempo CV: {t_dur:.2f}s")

    # Plot Comparação dos Modelos
    plt.figure(figsize=(8, 5), dpi=300)
    box_data = [cv_results[name] * 100 for name in classifiers]
    box = plt.boxplot(box_data, patch_artist=True, labels=list(classifiers.keys()))
    
    colors = ["#4575b4", "#74add1", "#abd9e9"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for median in box["medians"]:
        median.set(color="red", linewidth=2)

    plt.title("Comparativo de Acurácia dos Classificadores (5-Fold Stratified CV)", fontsize=12, fontweight="bold", pad=12)
    plt.ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
    plt.ylim(90, 101)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    comp_path = figures_dir / "classifier_comparison.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    logger.info(f"Salvo comparativo em: {comp_path}")

    # Selecionar o melhor modelo
    best_name = max(cv_results, key=lambda k: cv_results[k].mean())
    best_model = classifiers[best_name]
    logger.info(f"Melhor modelo selecionado: {best_name} ({cv_results[best_name].mean()*100:.2f}%)")

    # Treinar no dataset total para matriz de confusão e métricas detalhadas
    best_model.fit(features, labels)
    preds = best_model.predict(features)
    acc = accuracy_score(labels, preds)

    report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)
    report_text = classification_report(labels, preds, target_names=class_names, digits=4)
    logger.info(f"\nRelatório de Classificação ({best_name}):\n{report_text}")

    # Plot Matriz de Confusão Normalizada
    cm_norm = confusion_matrix(labels, preds, normalize="true")
    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    plt.title(f"Matriz de Confusão Normalizada — {best_name}", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Classe Predita", fontsize=11, fontweight="bold")
    plt.ylabel("Classe Real", fontsize=11, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"Salva matriz de confusão em: {cm_path}")

    # Plot Métricas por Classe (Precision, Recall, F1)
    precisions = [report_dict[c]["precision"] * 100 for c in class_names]
    recalls = [report_dict[c]["recall"] * 100 for c in class_names]
    f1s = [report_dict[c]["f1-score"] * 100 for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(11, 5), dpi=300)
    plt.bar(x - width, precisions, width, label="Precision", color="#3182bd", edgecolor="black", linewidth=0.5)
    plt.bar(x, recalls, width, label="Recall", color="#6baed6", edgecolor="black", linewidth=0.5)
    plt.bar(x + width, f1s, width, label="F1-Score", color="#9ecae1", edgecolor="black", linewidth=0.5)

    plt.xticks(x, class_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=11, fontweight="bold")
    plt.title(f"Desempenho por Equipamento — {best_name}", fontsize=13, fontweight="bold", pad=15)
    plt.ylim(85, 102)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    metrics_path = figures_dir / "metrics_per_class.png"
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    logger.info(f"Salvo gráfico de métricas por classe em: {metrics_path}")

    # Salvar artefato do modelo
    model_payload = {
        "model_name": best_name,
        "pipeline": best_model,
        "classes": class_names,
        "cv_accuracy_mean": float(cv_results[best_name].mean()),
        "cv_accuracy_std": float(cv_results[best_name].std()),
        "report": report_dict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(models_dir / "classifier_dinov2.pkl", "wb") as f:
        pickle.dump(model_payload, f)

    with open(models_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "cv_accuracy_mean": float(cv_results[best_name].mean()),
            "cv_accuracy_std": float(cv_results[best_name].std()),
            "overall_accuracy": float(acc),
            "cv_results": {k: [float(v) for v in vals] for k, vals in cv_results.items()},
            "per_class": {c: report_dict[c] for c in class_names}
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Salvo artefatos do modelo em {models_dir}/")
    return best_name, cv_results, report_dict, acc


def main():
    logger.info("Iniciando execução integrada do Pipeline DINOv2.")
    figures_dir, processed_dir, features_dir, models_dir = setup_directories()

    config = DINOv2Config(
        data_dir=Path("data/raw"),
        image_size=224,
        batch_size=32,
        num_workers=0,
        device="cuda"  # fallback automático para cpu se cuda indisponível
    )

    dataset, class_names = step1_eda_and_splits(config, figures_dir, processed_dir)
    features, labels, paths, elapsed_feat = step2_extract_features(config, dataset, features_dir)
    step3_tsne_projection(features, labels, class_names, figures_dir)
    best_name, cv_results, report, acc = step4_train_and_evaluate(features, labels, class_names, figures_dir, models_dir)

    logger.info("=== PIPELINE CONCLUÍDO COM SUCESSO! ===")
    logger.info(f"Melhor Modelo: {best_name}")
    logger.info(f"Acurácia Média 5-Fold: {cv_results[best_name].mean()*100:.2f}% ± {cv_results[best_name].std()*100:.2f}%")


if __name__ == "__main__":
    main()
