from google.colab import files
uploaded = files.upload()

import pandas as pd
csv_path = next(iter(uploaded.keys()))
df = pd.read_csv(csv_path)
print("Loaded:", csv_path)
print("Shape:", df.shape)
df.head(3)

import numpy as np
import re

df.columns = [c.strip() for c in df.columns]

def find_col(candidates):
    cand_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cand_lower:
            return cand_lower[name.lower()]
    return None

assembly_col = find_col(["Assembly", "GenomeAssembly", "Genome_Assembly"])
clin_sig_col = find_col(["ClinicalSignificance", "Clinical_Significance", "ClinSig"])

if assembly_col is None:
    raise ValueError("Assembly column not found. Columns:\n" + "\n".join(df.columns))
if clin_sig_col is None:
    raise ValueError("ClinicalSignificance column not found. Columns:\n" + "\n".join(df.columns))

print("Assembly col:", assembly_col)
print("ClinicalSignificance col:", clin_sig_col)

df = df[df[assembly_col].astype(str).str.upper().str.contains("GRCH38", na=False)].copy()
print("After GRCh38 filter:", df.shape)

key_cols = [c for c in ["GeneSymbol", "Chromosome", "Start", "Stop", "ReferenceAllele", "AlternateAllele"] if c in df.columns]
if len(key_cols) >= 4:
    df["__variant_key__"] = df[key_cols].astype(str).agg("|".join, axis=1)
    before = df.shape[0]
    df = df.drop_duplicates("__variant_key__").copy()
    print(f"Deduplicated by {key_cols}: {before} -> {df.shape[0]}")
else:
    print("Warning: Not enough key columns found for de-duplication:", key_cols)

def normalize_terms(s: str):
    if pd.isna(s):
        return []
    s = str(s).lower()
    parts = re.split(r"[;/|,]+", s)
    parts = [re.sub(r"\s+", " ", p.strip()) for p in parts if p.strip()]
    return parts
PATH_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic", "likely pathogenic/pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign", "likely benign/benign"}
EXCLUDE_SUBSTRINGS = ["uncertain", "vus", "conflicting", "not provided", "other", "risk factor",
                      "association", "drug response", "protective", "affects", "-"]
def map_label(clin_sig):
    terms = normalize_terms(clin_sig)
    if not terms:
        return np.nan
    joined = " | ".join(terms)
    has_path = any(t in PATH_TERMS for t in terms) or ("pathogenic" in joined and "benign" not in joined)
    has_ben  = any(t in BENIGN_TERMS for t in terms) or ("benign" in joined and "pathogenic" not in joined)
    if any(x in joined for x in EXCLUDE_SUBSTRINGS):
        pass
    if has_path and not has_ben:
        if any(x in joined for x in ["conflicting", "uncertain", "vus", "not provided"]):
            return np.nan
        return 1
    if has_ben and not has_path:
        if any(x in joined for x in ["conflicting", "uncertain", "vus", "not provided"]):
            return np.nan
        return 0
    return np.nan
df["y_pathogenic"] = df[clin_sig_col].apply(map_label)
before = df.shape[0]
df = df.dropna(subset=["y_pathogenic"]).copy()
df["y_pathogenic"] = df["y_pathogenic"].astype(int)

print(f"After label filter: {before} -> {df.shape[0]}")
print(df["y_pathogenic"].value_counts())

leakage_like = {
    "ReviewStatus", "NumberSubmitters", "Guidelines", "LastEvaluated", "ClinSigSimple",
    "AssertionCriteria", "ClinicalSignificance", "Clinical_Significance", "ClinSig",
    "RCVaccession", "SCVaccession", "VariationID", "AlleleID", "ClinVarAccession",
    "CLNHGVS", "HGVS", "HGVS_c", "HGVS_p", "ProteinChange",
    "__variant_key__"
}

cols_lower = {c.lower(): c for c in df.columns}
to_drop = []
for c in leakage_like:
    if c.lower() in cols_lower:
        to_drop.append(cols_lower[c.lower()])

for c in df.columns:
    cl = c.lower()
    if any(p in cl for p in ["accession", "variationid", "alleleid", "rcv", "scv", "clinvar"]):
        to_drop.append(c)

to_drop = sorted(set([c for c in to_drop if c != "y_pathogenic"]))
print("Dropping:", to_drop)

df_clean = df.drop(columns=to_drop, errors="ignore").copy()
print("Cleaned shape:", df_clean.shape)
df_clean.head(3)

import numpy as np
import pandas as pd
import re
df = df_clean.copy()
drop_now = [
    "Assembly", "ReferenceAllele", "AlternateAllele", "TestedInGTR",
    "Name", "OtherIDs", "RS# (dbSNP)", "nsv/esv (dbVar)",
    "PhenotypeIDS", "PhenotypeList",
    "SomaticClinicalImpact", "SomaticClinicalImpactLastEvaluated", "ReviewStatusClinicalImpact",
    "Oncogenicity", "OncogenicityLastEvaluated", "ReviewStatusOncogenicity",
    "SubmitterCategories",
]

df = df.drop(columns=[c for c in drop_now if c in df.columns], errors="ignore")

print("After dropping risky/constant columns:", df.shape)
print("Remaining columns:", df.columns.tolist())

for col in ["ReferenceAlleleVCF", "AlternateAlleleVCF"]:
    if col not in df.columns:
        raise ValueError(f"Missing expected column: {col}")

df["ref_len"] = df["ReferenceAlleleVCF"].astype(str).str.len()
df["alt_len"] = df["AlternateAlleleVCF"].astype(str).str.len()
df["len_diff"] = df["alt_len"] - df["ref_len"]
df["abs_len_diff"] = df["len_diff"].abs()

df["is_snv"] = ((df["ref_len"] == 1) & (df["alt_len"] == 1)).astype(int)
df["is_insertion"] = (df["len_diff"] > 0).astype(int)
df["is_deletion"] = (df["len_diff"] < 0).astype(int)
if "Start" in df.columns and "Stop" in df.columns:
    df["span_len"] = (df["Stop"] - df["Start"] + 1).clip(lower=1)
if "OriginSimple" in df.columns and "Origin" in df.columns:
    df = df.drop(columns=["Origin"], errors="ignore")
df = df.drop(columns=["ReferenceAlleleVCF", "AlternateAlleleVCF"], errors="ignore")

print("After feature engineering:", df.shape)
df.head(3)

if "y_pathogenic" not in df.columns:
    raise ValueError("y_pathogenic not found.")

X = df.drop(columns=["y_pathogenic"])
y = df["y_pathogenic"].astype(int)

print("X shape:", X.shape, "y shape:", y.shape)
print("\nCategorical columns:", [c for c in X.columns if X[c].dtype == "object"])
print("Numeric columns:", [c for c in X.columns if X[c].dtype != "object"])
out_path = "/content/final_features_grch38_brca_binary.csv"
pd.concat([X, y], axis=1).to_csv(out_path, index=False)
print("Saved:", out_path)

df = df.drop(columns=["Start", "Stop"], errors="ignore")
print(df.shape)

import pandas as pd
X = df.drop(columns=["y_pathogenic"])
y = df["y_pathogenic"].astype(int)

print("X shape:", X.shape)
print("y distribution:\n", y.value_counts(normalize=True).round(3))

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ),
    "SVM_RBF": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}

from sklearn.metrics import roc_auc_score, average_precision_score

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)

    results.append({
        "Model": name,
        "ROC_AUC": round(roc, 3),
        "PR_AUC": round(pr, 3)
    })

pd.DataFrame(results)

!pip -q install shap

import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
np.random.seed(42)

pipes = {}
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pipes[name] = pipe

list(pipes.keys())

preprocess.fit(X_train)

X_train_t = preprocess.transform(X_train)
X_test_t  = preprocess.transform(X_test)
def get_feature_names_from_preprocess(preprocess, num_cols, cat_cols):
    feature_names = []
    feature_names += list(num_cols)
    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    cat_features = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names += cat_features
    return feature_names

feature_names = get_feature_names_from_preprocess(preprocess, num_cols, cat_cols)

X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
X_test_df  = pd.DataFrame(X_test_t, columns=feature_names)

print("Transformed shapes:", X_train_df.shape, X_test_df.shape)
X_train_df.head(2)

tree_models = ["RandomForest", "XGBoost"]
tree_shap_values = {}

for name in tree_models:
    model = pipes[name].named_steps["model"]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test_df)
    if isinstance(sv, list):
        sv = sv[1]

    tree_shap_values[name] = (explainer, sv)

    print(name, "SHAP shape:", sv.shape)

kernel_models = ["LogisticRegression", "SVM_RBF"]
kernel_shap_values = {}
bg_size = 200
bg = X_train_df.sample(n=min(bg_size, len(X_train_df)), random_state=42)
n_explain = 100
X_explain = X_test_df.sample(n=min(n_explain, len(X_test_df)), random_state=42)
for name in kernel_models:
    pipe = pipes[name]
    model = pipe.named_steps["model"]
    def predict_proba_transformed(X):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return model.predict_proba(X_arr)[:, 1]
    explainer = shap.KernelExplainer(predict_proba_transformed, bg)
    sv = explainer.shap_values(X_explain, nsamples=100)
    kernel_shap_values[name] = (explainer, X_explain, sv)
    print(name, "SHAP shape:", np.array(sv).shape)

import matplotlib.pyplot as plt
for name in tree_models:
    explainer, sv = tree_shap_values[name]
    shap.summary_plot(sv, X_test_df, show=False, max_display=20)
    plt.title(f"SHAP Summary (TreeSHAP) - {name}")
    plt.show()

import matplotlib.pyplot as plt
for name in kernel_models:
    explainer, X_explain, sv = kernel_shap_values[name]
    shap.summary_plot(sv, X_explain, show=False, max_display=20)
    plt.title(f"SHAP Summary (KernelSHAP) - {name}")
    plt.show()

def top_k_from_shap(shap_values, feature_names, k=20):
        shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    elif shap_values.ndim == 2:
        mean_abs = np.abs(shap_values).mean(axis=0)
    else:
        raise ValueError(f"Unexpected SHAP values dimensionality: {shap_values.ndim}")

    idx = np.argsort(mean_abs)[::-1][:k]
    return pd.DataFrame({
        "feature": np.array(feature_names)[idx],
        "mean_abs_shap": mean_abs[idx]
    })
for name in tree_models:
    _, sv = tree_shap_values[name]
    top = top_k_from_shap(sv, X_test_df.columns, k=20)
    print("\nTop 20 features:", name)
    display(top)
for name in kernel_models:
    _, X_explain, sv = kernel_shap_values[name]
    top = top_k_from_shap(np.array(sv), X_explain.columns, k=20)
    print("\nTop 20 features:", name)
    display(top)

out_dir = "/content/shap_outputs"
import os
os.makedirs(out_dir, exist_ok=True)
for name in tree_models:
    _, sv = tree_shap_values[name]
    top = top_k_from_shap(sv, X_test_df.columns, k=50)
    top.to_csv(f"{out_dir}/top50_{name}_TreeSHAP.csv", index=False)
for name in kernel_models:
    _, X_explain, sv = kernel_shap_values[name]
    top = top_k_from_shap(np.array(sv), X_explain.columns, k=50)
    top.to_csv(f"{out_dir}/top50_{name}_KernelSHAP.csv", index=False)

print("Saved to:", out_dir)

!pip -q install lime

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
bg = X_train_df.sample(n=min(2000, len(X_train_df)), random_state=42).values
feature_names_lime = X_train_df.columns.tolist()

explainer_lime = LimeTabularExplainer(
    training_data=bg,
    feature_names=feature_names_lime,
    class_names=["Benign", "Pathogenic"],
    mode="classification",
    discretize_continuous=True
)

def proba_from_pipe_transformed(pipe, X_transformed_df):
    model = pipe.named_steps["model"]
    X_arr = X_transformed_df.values if hasattr(X_transformed_df, "values") else X_transformed_df
    return model.predict_proba(X_arr)
chosen = "XGBoost"
pipe = pipes[chosen]

idx = 0
x_row = X_test_df.iloc[idx].values

exp = explainer_lime.explain_instance(
    data_row=x_row,
    predict_fn=lambda X: proba_from_pipe_transformed(pipe, pd.DataFrame(X, columns=feature_names_lime)),
    num_features=15
)
print("LIME explanation (top 15):")
for f, w in exp.as_list():
    print(f"{f}: {w:.4f}")

import matplotlib.pyplot as plt

fig = exp.as_pyplot_figure()
plt.title(f"LIME Local Explanation - {chosen} - test idx {idx}")
plt.savefig(f"lime_local_{chosen}_idx{idx}.png")
plt.show()

from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
def pr_auc_scorer(estimator, X, y):
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y, y_pred_proba)

perm_results = {}

for name, pipe in pipes.items():
    r = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring=pr_auc_scorer,
        n_jobs=-1
    )
    perm_results[name] = r
    print(name, "Permutation importance computed.")

from sklearn.base import clone

perm_transformed = {}
for name in pipes.keys():
    model = pipes[name].named_steps["model"]
    X_eval = X_test_df.copy()
    y_eval = y_test.values

    def pr_auc_model(model, X, y):
        return average_precision_score(y, model.predict_proba(X)[:, 1])
    base_score = pr_auc_model(model, X_eval.values, y_eval)

    rng = np.random.RandomState(42)
    importances = np.zeros(X_eval.shape[1])

    for j, col in enumerate(X_eval.columns):
        scores = []
        for _ in range(10):
            Xp = X_eval.values.copy()
            rng.shuffle(Xp[:, j])
            scores.append(pr_auc_model(model, Xp, y_eval))
        importances[j] = base_score - np.mean(scores)

    perm_transformed[name] = pd.Series(importances, index=X_eval.columns).sort_values(ascending=False)
    print("Permutation (transformed) done:", name)

display(perm_transformed["XGBoost"].head(20).to_frame("perm_drop_in_PR_AUC"))

builtin_importance = {}
if "LogisticRegression" in pipes:
    lr = pipes["LogisticRegression"].named_steps["model"]
    coef = pd.Series(lr.coef_.ravel(), index=X_train_df.columns)
    builtin_importance["LogisticRegression_coef_abs"] = coef.abs().sort_values(ascending=False)
for name in ["RandomForest", "XGBoost"]:
    if name in pipes:
        m = pipes[name].named_steps["model"]
        fi = pd.Series(m.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)
        builtin_importance[f"{name}_feature_importances"] = fi

for k, v in builtin_importance.items():
    print("\nTop 15 built-in:", k)
    display(v.head(15).to_frame("importance"))

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import shap
def get_feature_names_from_preprocess(preprocess_obj, num_cols, cat_cols):
    feature_names = []
    feature_names += list(num_cols)
    ohe = preprocess_obj.named_transformers_["cat"].named_steps["onehot"]
    cat_features = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names += cat_features
    return feature_names

def mean_abs_shap_tree(model, X_df):
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X_df, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]
    return np.abs(sv).mean(axis=0)

def builtin_vector(model):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    return None

def lr_coef_abs(model):
    return np.abs(model.coef_.ravel())


B = 20
n_shap = 400
rng = np.random.RandomState(42)
preprocess.fit(X_train)
fixed_feature_names = get_feature_names_from_preprocess(preprocess, num_cols, cat_cols)
idx_shap = rng.choice(len(X_test), size=min(n_shap, len(X_test)), replace=False)
X_test_small = X_test.iloc[idx_shap].copy()
X_test_small_t = preprocess.transform(X_test_small)
X_test_small_df = pd.DataFrame(X_test_small_t, columns=fixed_feature_names)

bootstrap_store = {
    "RandomForest": {"shap": [], "builtin": []},
    "XGBoost": {"shap": [], "builtin": []},
    "LogisticRegression": {"coef": []}
}
rf_base = clone(models["RandomForest"]).set_params(n_estimators=150, n_jobs=-1)
xgb_base = clone(models["XGBoost"]).set_params(n_estimators=150)

for b in range(B):
    Xb, yb = resample(
        X_train, y_train,
        replace=True,
        random_state=1000 + b,
        stratify=y_train
    )
    Xb_transformed = preprocess.transform(Xb)
    Xb_df_transformed = pd.DataFrame(Xb_transformed, columns=fixed_feature_names)
    rf = clone(rf_base).set_params(random_state=1000 + b)
    rf.fit(Xb_df_transformed, yb)

    bootstrap_store["RandomForest"]["builtin"].append(builtin_vector(rf))
    bootstrap_store["RandomForest"]["shap"].append(mean_abs_shap_tree(rf, X_test_small_df))
    xgb = clone(xgb_base).set_params(random_state=2000 + b)
    xgb.fit(Xb_df_transformed, yb)

    bootstrap_store["XGBoost"]["builtin"].append(builtin_vector(xgb))
    bootstrap_store["XGBoost"]["shap"].append(mean_abs_shap_tree(xgb, X_test_small_df))
    lr = clone(models["LogisticRegression"]).set_params(random_state=3000 + b)
    lr.fit(Xb_df_transformed, yb)

    bootstrap_store["LogisticRegression"]["coef"].append(lr_coef_abs(lr))

    if (b + 1) % 5 == 0:
        print(f"Bootstrap {b+1}/{B} done")

for m in ["RandomForest", "XGBoost"]:
    for method in ["shap", "builtin"]:
        bootstrap_store[m][method] = np.vstack(bootstrap_store[m][method])
bootstrap_store["LogisticRegression"]["coef"] = np.vstack(bootstrap_store["LogisticRegression"]["coef"])

print("RF shap:", bootstrap_store["RandomForest"]["shap"].shape)
print("XGB shap:", bootstrap_store["XGBoost"]["shap"].shape)
print("LR coef:", bootstrap_store["LogisticRegression"]["coef"].shape)

from scipy.stats import spearmanr
def topk_idx(v, k):
    return np.argsort(v)[::-1][:k]
def jaccard_topk(v1, v2, k):
    a = set(topk_idx(v1, k))
    b = set(topk_idx(v2, k))
    return len(a & b) / max(1, len(a | b))
def stability_report(V, k=10):
    B = V.shape[0]
    spears = []
    jacs = []
    for i in range(B):
        for j in range(i+1, B):
            spears.append(spearmanr(V[i], V[j]).correlation)
            jacs.append(jaccard_topk(V[i], V[j], k))
    return float(np.nanmean(spears)), float(np.nanmean(jacs))

stability_table = []

for model_name in ["RandomForest", "XGBoost"]:
    for method in ["shap", "builtin"]:
        V = bootstrap_store[model_name][method]
        sp, jc = stability_report(V, k=top_k)
        stability_table.append([model_name, method, sp, jc])
sp, jc = stability_report(bootstrap_store["LogisticRegression"]["coef"], k=top_k)
stability_table.append(["LogisticRegression", "coef_abs", sp, jc])

stability_df = pd.DataFrame(stability_table, columns=["Model", "Method", "Mean_Spearman", f"Mean_Jaccard@{top_k}"])
stability_df

import matplotlib.pyplot as plt

def stability_heatmap(V, title, fname):
    B = V.shape[0]
    corr = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            corr[i, j] = spearmanr(V[i], V[j]).correlation
    plt.figure()
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Bootstrap run")
    plt.ylabel("Bootstrap run")
    plt.show()

stability_heatmap(bootstrap_store["XGBoost"]["shap"],
                  "Stability Heatmap: Spearman(mean|SHAP|) - XGBoost",
                  "stab_heatmap_xgb_shap")

def FSI(mean_spearman, mean_jaccard, w_spear=0.6, w_jacc=0.4):
    # Spearman ranges [-1,1] -> map to [0,1]
    spear01 = (mean_spearman + 1) / 2
    jacc01 = mean_jaccard  # already [0,1]
    return w_spear * spear01 + w_jacc * jacc01

stability_df["FSI"] = stability_df.apply(lambda r: FSI(r["Mean_Spearman"], r[f"Mean_Jaccard@{top_k}"]), axis=1)
stability_df.sort_values("FSI", ascending=False)

out_path = "/content/stability_fsi_table.csv"
stability_df.to_csv(out_path, index=False)
out_path

def normalize_vector(v):
    v = np.array(v, dtype=float)
    v = np.abs(v)
    s = v.sum()
    return v / s if s > 0 else v

# XGBoost vectors:
xgb_shap = np.abs(tree_shap_values["XGBoost"][1]).mean(axis=0)           # mean|SHAP|
xgb_builtin = builtin_importance["XGBoost_feature_importances"].values    # built-in
xgb_perm = perm_transformed["XGBoost"].reindex(X_test_df.columns).fillna(0).values  # perm drop in PR-AUC

methods = {
    "SHAP": normalize_vector(xgb_shap),
    "BuiltIn": normalize_vector(xgb_builtin),
    "Permutation": normalize_vector(xgb_perm),
}

# Pairwise Spearman + Jaccard
names = list(methods.keys())
S = np.zeros((len(names), len(names)))
J = np.zeros((len(names), len(names)))

for i, a in enumerate(names):
    for j, b in enumerate(names):
        S[i, j] = spearmanr(methods[a], methods[b]).correlation
        J[i, j] = jaccard_topk(methods[a], methods[b], top_k)

print("Spearman matrix:\n", pd.DataFrame(S, index=names, columns=names).round(3))
print(f"\nJaccard@{top_k} matrix:\n", pd.DataFrame(J, index=names, columns=names).round(3))

import matplotlib.pyplot as plt

def matrix_plot(M, labels, title, fname):
    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.show()

matrix_plot(S, names, "Cross-method Agreement (Spearman) - XGBoost", "cross_method_spearman_xgb")
matrix_plot(J, names, f"Cross-method Top-{top_k} Overlap (Jaccard) - XGBoost", f"cross_method_jaccard_top{top_k}_xgb")

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

plt.figure()
for name, pipe in pipes.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (All Models)")
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

plt.figure()
for name, pipe in pipes.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    p, r, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    plt.plot(r, p, label=f"{name} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (All Models)")
plt.legend()
plt.show()

import shap
import numpy as np

_, sv_xgb = tree_shap_values["XGBoost"]
if isinstance(sv_xgb, np.ndarray) and sv_xgb.ndim == 3:
    sv_xgb = sv_xgb[:, :, 1]

shap.summary_plot(sv_xgb, X_test_df, max_display=20)

mean_abs = np.abs(sv_xgb).mean(axis=0)
top_feature = X_test_df.columns[int(np.argmax(mean_abs))]
print("Top SHAP feature:", top_feature)

shap.dependence_plot(top_feature, sv_xgb, X_test_df)

explainer_xgb = tree_shap_values["XGBoost"][0]
exp = explainer_xgb(X_test_df)

xgb_model = pipes["XGBoost"].named_steps["model"]
y_prob = xgb_model.predict_proba(X_test_df.values)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

tp_idx = np.where((y_test.values == 1) & (y_pred == 1))[0][0]
print("TP index:", tp_idx, "Predicted prob:", y_prob[tp_idx])

shap.plots.waterfall(exp[tp_idx], max_display=15)

from scipy.stats import spearmanr

V = bootstrap_store["XGBoost"]["shap"]  # (B, n_features)
B = V.shape[0]

corr = np.zeros((B, B))
for i in range(B):
    for j in range(B):
        corr[i, j] = spearmanr(V[i], V[j]).correlation

plt.figure()
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.xlabel("Bootstrap run")
plt.ylabel("Bootstrap run")
plt.title("Stability Heatmap — Spearman(mean|SHAP|) (XGBoost)")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

display(stability_df.sort_values("FSI", ascending=False))

plt.figure(figsize=(10, 4))
plt.axis("off")
tbl = plt.table(
    cellText=stability_df.round(3).values,
    colLabels=stability_df.columns.tolist(),
    cellLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.25)
plt.title("FSI Comparison Table (Models × Methods)")
plt.show()

from scipy.stats import spearmanr

def normalize(v):
    v = np.abs(np.array(v))
    return v / v.sum() if v.sum() > 0 else v

v_shap = normalize(np.abs(sv_xgb).mean(axis=0))
v_builtin = normalize(
    builtin_importance["XGBoost_feature_importances"]
    .reindex(X_test_df.columns)
    .fillna(0)
    .values
)
v_perm = normalize(
    perm_transformed["XGBoost"]
    .reindex(X_test_df.columns)
    .fillna(0)
    .values
)

labels = ["SHAP", "Permutation", "Built-in"]
vectors = [v_shap, v_perm, v_builtin]

M = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        M[i, j] = spearmanr(vectors[i], vectors[j]).correlation

plt.figure()
plt.imshow(M, aspect="auto")
plt.colorbar()
plt.xticks(range(3), labels)
plt.yticks(range(3), labels)
plt.title("Cross-method Agreement (Spearman) — XGBoost")
plt.show()
