#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=======================================================================
Pipeline de Detecção de Fraudes — PaySim
TCC: Arquitetura Orientada a Eventos (EDA) com IA para Anti-Fraude
Felipe Pucci Veloso e Túlio Teixeira Silva — Uni-FACEF
=======================================================================

O QUE ESTE SCRIPT FAZ (na ordem):
  1. Carrega o CSV do PaySim.
  2. Análise Exploratória (EDA): distribuição de classes, tipos, saldos.
  3. Filtra os tipos onde há fraude (TRANSFER e CASH_OUT).
  4. Engenharia de atributos: erros de saldo de origem e destino.
  5. Codificação, split treino/teste estratificado.
  6. Tratamento de desbalanceamento (class_weight e, opcionalmente, SMOTE).
  7. Treina 3 modelos: Regressão Logística, Árvore de Decisão, Random Forest.
  8. Avalia: precisão, recall, F1, AUC-ROC, AUC-PR, matriz de confusão.
  9. Salva: tabela de resultados (CSV), matrizes de confusão e figuras (PNG).

COMO RODAR:
  python fraud_pipeline.py --csv PS_20174392719_1491204439457_log.csv
  (use --smote para ativar o SMOTE; --sample 0.2 para testar com 20% dos dados)

SAÍDAS geradas na pasta ./resultados/:
  - tabela_resultados.csv          (a Tabela 6 do artigo)
  - matriz_confusao_<modelo>.png
  - importancia_atributos.png
  - distribuicao_classes.png / fraude_por_tipo.png
=======================================================================
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# paleta (mesma do artigo/slides)
NAVY = "#16285C"
CYAN = "#16C0D6"
OUTDIR = "resultados"


# ----------------------------------------------------------------------
# 1. CARGA
# ----------------------------------------------------------------------
def carregar_dados(caminho_csv, sample_frac=None, seed=42):
    print(f"[1] Carregando dados de: {caminho_csv}")
    df = pd.read_csv(caminho_csv)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
        print(f"    (amostra de {sample_frac:.0%} -> {len(df):,} linhas)")
    print(f"    Linhas: {len(df):,} | Colunas: {list(df.columns)}")
    return df


# ----------------------------------------------------------------------
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ----------------------------------------------------------------------
def eda(df):
    print("\n[2] Análise Exploratória (EDA)")
    total = len(df)
    n_fraude = int(df["isFraud"].sum())
    print(f"    Total de transações : {total:,}")
    print(f"    Fraudes             : {n_fraude:,} ({100*n_fraude/total:.4f}%)")
    print("    Transações por tipo :")
    print(df["type"].value_counts().to_string())
    print("    Fraudes por tipo    :")
    print(df[df["isFraud"] == 1]["type"].value_counts().to_string())

    # Figura: distribuição de classes
    plt.figure(figsize=(5, 4))
    vc = df["isFraud"].value_counts().sort_index()
    plt.bar(["Legítima", "Fraude"], vc.values, color=[NAVY, CYAN])
    for i, v in enumerate(vc.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.title("Distribuição de classes (isFraud)")
    plt.ylabel("Nº de transações"); plt.yscale("log")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/distribuicao_classes.png", dpi=140); plt.close()

    # Figura: fraude por tipo
    plt.figure(figsize=(6, 4))
    piv = df.groupby("type")["isFraud"].sum().sort_values(ascending=False)
    plt.bar(piv.index, piv.values, color=NAVY)
    plt.title("Nº de fraudes por tipo de transação")
    plt.ylabel("Nº de fraudes"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fraude_por_tipo.png", dpi=140); plt.close()
    print(f"    Figuras salvas em ./{OUTDIR}/")


# ----------------------------------------------------------------------
# 3-4. PRÉ-PROCESSAMENTO E ENGENHARIA DE ATRIBUTOS
# ----------------------------------------------------------------------
def preprocessar(df):
    print("\n[3-4] Pré-processamento e engenharia de atributos")
    # Fraude ocorre apenas em TRANSFER e CASH_OUT -> restringe o escopo
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    print(f"    Após filtrar TRANSFER/CASH_OUT: {len(df):,} linhas")

    # Engenharia: erros de saldo (altamente preditivos)
    df["erroSaldoOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["erroSaldoDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Codifica o tipo (TRANSFER=0, CASH_OUT=1)
    df["type_cod"] = (df["type"] == "CASH_OUT").astype(int)

    # Remove identificadores e colunas não usadas
    features = ["type_cod", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "erroSaldoOrig", "erroSaldoDest", "step"]
    X = df[features]
    y = df["isFraud"].astype(int)
    print(f"    Atributos usados: {features}")
    return X, y, features


# ----------------------------------------------------------------------
# 5. SPLIT
# ----------------------------------------------------------------------
def dividir(X, y, seed=42):
    print("\n[5] Split treino/teste (estratificado 70/30)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    print(f"    Treino: {len(X_tr):,} | Teste: {len(X_te):,}")
    print(f"    Fraudes no teste: {int(y_te.sum()):,}")
    return X_tr, X_te, y_tr, y_te


# ----------------------------------------------------------------------
# 6-7-8. TREINO E AVALIAÇÃO
# ----------------------------------------------------------------------
def treinar_avaliar(X_tr, X_te, y_tr, y_te, features, usar_smote=False, seed=42):
    print("\n[6-7-8] Treinamento e avaliação dos modelos")

    # Escala (necessária p/ Regressão Logística)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Tratamento de desbalanceamento
    if usar_smote:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=seed)
        X_tr_s, y_tr_bal = sm.fit_resample(X_tr_s, y_tr)
        X_tr_rf, y_tr_rf = sm.fit_resample(X_tr, y_tr)
        print(f"    SMOTE aplicado -> treino balanceado: {len(y_tr_bal):,}")
    else:
        y_tr_bal = y_tr
        X_tr_rf, y_tr_rf = X_tr, y_tr

    modelos = {
        "Regressão Logística": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            X_tr_s if usar_smote else X_tr_s, X_te_s, y_tr_bal if usar_smote else y_tr),
        "Árvore de Decisão": (
            DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=seed),
            X_tr_rf, X_te, y_tr_rf),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=16, n_jobs=-1,
                                   class_weight="balanced", random_state=seed),
            X_tr_rf, X_te, y_tr_rf),
    }

    linhas = []
    rf_model = None
    for nome, (modelo, Xtr, Xte, ytr) in modelos.items():
        modelo.fit(Xtr, ytr)
        y_pred = modelo.predict(Xte)
        y_prob = modelo.predict_proba(Xte)[:, 1]

        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob)
        aucpr = average_precision_score(y_te, y_prob)
        linhas.append({
            "Modelo": nome, "Precisão": round(prec, 3), "Recall": round(rec, 3),
            "F1-score": round(f1, 3), "AUC-ROC": round(auc, 3), "AUC-PR": round(aucpr, 3)})
        print(f"    {nome:22s} | P={prec:.3f} R={rec:.3f} F1={f1:.3f} AUC={auc:.3f} AUC-PR={aucpr:.3f}")

        # matriz de confusão
        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(3.6, 3.2))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, f"{v:,}", ha="center", va="center",
                     color="white" if v > cm.max()/2 else "black", fontsize=10)
        plt.xticks([0, 1], ["Legítima", "Fraude"]); plt.yticks([0, 1], ["Legítima", "Fraude"])
        plt.xlabel("Predito"); plt.ylabel("Real"); plt.title(f"Matriz de Confusão\n{nome}")
        plt.tight_layout()
        fname = nome.lower().replace(" ", "_").replace("ç", "c").replace("ã", "a").replace("ó", "o").replace("á", "a")
        plt.savefig(f"{OUTDIR}/matriz_confusao_{fname}.png", dpi=140); plt.close()

        if nome == "Random Forest":
            rf_model = modelo

    # importância de atributos (Random Forest)
    if rf_model is not None:
        imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
        plt.figure(figsize=(6, 4))
        plt.barh(imp.index, imp.values, color=CYAN)
        plt.title("Importância dos atributos (Random Forest)")
        plt.tight_layout(); plt.savefig(f"{OUTDIR}/importancia_atributos.png", dpi=140); plt.close()

    resultados = pd.DataFrame(linhas)
    resultados.to_csv(f"{OUTDIR}/tabela_resultados.csv", index=False, encoding="utf-8-sig")
    print(f"\n[9] Resultados salvos em ./{OUTDIR}/tabela_resultados.csv")
    return resultados


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pipeline PaySim - detecção de fraude")
    ap.add_argument("--csv", required=True, help="Caminho do CSV do PaySim")
    ap.add_argument("--smote", action="store_true", help="Ativar SMOTE")
    ap.add_argument("--sample", type=float, default=None, help="Fração da base (ex.: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    df = carregar_dados(args.csv, args.sample, args.seed)
    eda(df)
    X, y, feats = preprocessar(df)
    X_tr, X_te, y_tr, y_te = dividir(X, y, args.seed)
    res = treinar_avaliar(X_tr, X_te, y_tr, y_te, feats, args.smote, args.seed)
    print("\n================= TABELA FINAL =================")
    print(res.to_string(index=False))
    print("===============================================")
    print("\nConcluído. Use os arquivos em ./resultados/ no seu artigo.")


if __name__ == "__main__":
    main()