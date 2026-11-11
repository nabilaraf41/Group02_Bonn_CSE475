"""CSE475 Task 2: baselines and temporal-graph GCN on Bonn EEG.

Uses the fixed Task 1 split (outputs/task1/split_manifest.csv): 300 train / 100 validation / 100 test.
All scalers, k choices and early stopping use train/validation only; the test set is touched once per model.
"""
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
                             classification_report)

CLASSES = ['Z', 'O', 'N', 'F', 'S']
N_NODES = 8
SEED = 475


def locate_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / 'data/manifest.json').exists() and (p / 'outputs/task1/split_manifest.csv').exists():
            return p
    raise FileNotFoundError('Run from inside the Group02_Bonn_CSE475 repository.')


def load(root):
    """Return signals, labels, splits for all 500 segments using the Task 1 split."""
    sys.path.insert(0, str(root / 'code/task1'))
    import eda
    meta = json.loads((root / 'data/metadata.json').read_text())
    split = pd.read_csv(root / 'outputs/task1/split_manifest.csv')
    X = np.stack([np.loadtxt(root / 'data' / p) for p in split.path])
    y = split.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    assert X.shape == (500, 4097) and np.isfinite(X).all()
    return X, y, split.split.to_numpy(), split.path.to_numpy(), meta['sampling_rate_hz'], eda.features


def segment_features(X, fs, features):
    return pd.DataFrame([features(x, fs) for x in X])


def node_features(X, fs, features):
    """8 contiguous temporal parts per segment (513 + 7x512), 19 descriptors each -> (n, 8, 19)."""
    return np.stack([[list(features(part, fs).values()) for part in np.array_split(x, N_NODES)] for x in X])


# ---------------- metrics ----------------
def evaluate(y, proba, name, train_s=None, infer_s=None, params=None):
    pred = proba.argmax(1)
    yb = label_binarize(y, classes=range(len(CLASSES)))
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)
    _, rec, _, _ = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    row = {'model': name, 'accuracy': accuracy_score(y, pred), 'precision_macro': p_m, 'recall_macro': r_m,
           'f1_macro': f_m, 'precision_weighted': p_w, 'recall_weighted': r_w, 'f1_weighted': f_w,
           'roc_auc_ovr_macro': roc_auc_score(yb, proba, average='macro'),
           'pr_auc_macro': average_precision_score(yb, proba, average='macro'),
           **{f'recall_{c}': v for c, v in zip(CLASSES, rec)}}
    if train_s is not None: row['train_time_s'] = train_s
    if infer_s is not None: row['inference_time_s'] = infer_s
    if params is not None: row['parameters'] = params
    return row


def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=range(5))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted', ylabel='True', title=title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc_pr(y, proba, title, path):
    yb = label_binarize(y, classes=range(5))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        a.plot(fpr, tpr, label=f'{c} (AUC {roc_auc_score(yb[:, i], proba[:, i]):.3f})')
        pr, rc, _ = precision_recall_curve(yb[:, i], proba[:, i])
        b.plot(rc, pr, label=f'{c} (AP {average_precision_score(yb[:, i], proba[:, i]):.3f})')
    a.plot([0, 1], [0, 1], 'k--', lw=.7)
    a.set(title=f'{title}: ROC (one-vs-rest)', xlabel='False positive rate', ylabel='True positive rate'); a.legend(fontsize=8)
    b.set(title=f'{title}: precision-recall', xlabel='Recall', ylabel='Precision'); b.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- baselines ----------------
def classical_models():
    import xgboost as xgb
    s = SEED
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=s)),
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(probability=True, random_state=s)),
        'k-NN (k=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        'Decision Tree': DecisionTreeClassifier(random_state=s),
        'Random Forest': RandomForestClassifier(500, random_state=s, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=s),
        'XGBoost': xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.05, random_state=s, n_jobs=-1),
        'MLP': make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=3000, early_stopping=True, random_state=s)),
    }


def run_classical(F, y, split):
    tr, te = split == 'train', split == 'test'
    rows, probas = [], {}
    for name, model in classical_models().items():
        t = time.perf_counter(); model.fit(F[tr], y[tr]); ts = time.perf_counter() - t
        t = time.perf_counter(); p = model.predict_proba(F[te]); ti = time.perf_counter() - t
        rows.append(evaluate(y[te], p, name, ts, ti)); probas[name] = p
    return pd.DataFrame(rows), probas


def train_cnn1d(X, y, split, seed=SEED, epochs=150, patience=25):
    """Small 1D-CNN on the raw signal. Amplitude scale fitted on training segments only."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    scale = X[split == 'train'].std()
    T = lambda m: torch.tensor(X[m] / scale, dtype=torch.float32).unsqueeze(1)
    xtr, xva, xte = T(split == 'train'), T(split == 'validation'), T(split == 'test')
    ytr = torch.tensor(y[split == 'train'])
    net = nn.Sequential(
        nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
        nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        nn.Flatten(), nn.Dropout(.3), nn.Linear(64, 5))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    best, best_state, wait = -1, None, 0
    t = time.perf_counter()
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(xtr)).split(32):
            opt.zero_grad(); nn.functional.cross_entropy(net(xtr[idx]), ytr[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            f = f1_score(y[split == 'validation'], net(xva).argmax(1).numpy(), average='macro')
        if f > best: best, best_state, wait = f, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    ts = time.perf_counter() - t
    net.load_state_dict(best_state); net.eval()
    t = time.perf_counter()
    with torch.no_grad(): p = torch.softmax(net(xte), 1).numpy()
    ti = time.perf_counter() - t
    params = sum(q.numel() for q in net.parameters())
    return evaluate(y[split == 'test'], p, '1D-CNN (raw signal)', ts, ti, params), p


# ---------------- proposed GCN ----------------
def build_graphs(NF, mode, k=2):
    """One graph per segment. Nodes = 8 temporal parts; NF already standardised with train-node statistics.
    mode: 'temporal' (chain i<->i+1), 'similarity' (k nearest nodes in feature space, same segment), 'union'.
    Returns normalised adjacency D^-1/2 (A+I) D^-1/2, shape (n, 8, 8), and undirected edge counts.
    Edges never cross segments, so no information moves between train/validation/test."""
    n = len(NF); A = np.zeros((n, N_NODES, N_NODES))
    if mode in ('temporal', 'union'):
        i = np.arange(N_NODES - 1); A[:, i, i + 1] = A[:, i + 1, i] = 1
    if mode in ('similarity', 'union'):
        d = ((NF[:, :, None] - NF[:, None]) ** 2).sum(-1)
        d[:, np.arange(N_NODES), np.arange(N_NODES)] = np.inf
        nn_idx = np.argsort(d, -1)[..., :k]
        for g in range(n):
            for a in range(N_NODES):
                A[g, a, nn_idx[g, a]] = A[g, nn_idx[g, a], a] = 1
    edges = A.sum((1, 2)) / 2
    A = A + np.eye(N_NODES)
    dinv = 1 / np.sqrt(A.sum(-1))
    return A * dinv[:, :, None] * dinv[:, None, :], edges


def make_gcn(in_dim, hidden=64, dropout=.3):
    import torch, torch.nn as nn

    class TemporalGCN(nn.Module):
        """2 GCN layers (H' = A_hat H W) -> mean+max readout over the 8 nodes -> linear classifier."""
        def __init__(self):
            super().__init__()
            self.w1, self.w2 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden)
            self.b1, self.b2 = nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(2 * hidden, 5)

        def layer(self, A, h, w, bn):
            h = A @ w(h)                                   # aggregate neighbours (incl. self)
            h = bn(h.reshape(-1, h.shape[-1])).reshape(h.shape)
            return self.drop(torch.relu(h))

        def forward(self, A, x):
            h = self.layer(A, x, self.w1, self.b1)
            h = self.layer(A, h, self.w2, self.b2)
            return self.head(torch.cat([h.mean(1), h.amax(1)], 1))
    return TemporalGCN()


def train_gcn(NF, y, split, mode, k=2, seed=SEED, epochs=300, patience=40):
    import torch, torch.nn.functional as Fn
    torch.manual_seed(seed); np.random.seed(seed)
    tr = split == 'train'
    scaler = StandardScaler().fit(NF[tr].reshape(-1, NF.shape[-1]))  # train nodes only
    Z = scaler.transform(NF.reshape(-1, NF.shape[-1])).reshape(NF.shape)
    A, edges = build_graphs(Z, mode, k)
    T = lambda m: (torch.tensor(A[m], dtype=torch.float32), torch.tensor(Z[m], dtype=torch.float32))
    (atr, xtr), va, te = T(split == 'train'), T(split == 'validation'), T(split == 'test')
    ytr = torch.tensor(y[split == 'train'])
    net = make_gcn(NF.shape[-1])
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=5e-4)
    best, best_state, wait, best_ep = -1, None, 0, 0
    t = time.perf_counter()
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(xtr)).split(32):
            opt.zero_grad(); Fn.cross_entropy(net(atr[idx], xtr[idx]), ytr[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            f = f1_score(y[split == 'validation'], net(*va).argmax(1).numpy(), average='macro')
        if f > best: best, best_state, wait, best_ep = f, {kk: v.clone() for kk, v in net.state_dict().items()}, 0, ep
        else:
            wait += 1
            if wait >= patience: break
    ts = time.perf_counter() - t
    net.load_state_dict(best_state); net.eval()
    t = time.perf_counter()
    with torch.no_grad(): p = torch.softmax(net(*te), 1).numpy()
    ti = time.perf_counter() - t
    params = sum(q.numel() for q in net.parameters())
    info = {'val_macro_f1': best, 'best_epoch': best_ep, 'n_edges_mean': float(edges[split == 'test'].mean())}
    return evaluate(y[split == 'test'], p, f'GCN ({mode})', ts, ti, params), p, info, net


def choose_k(NF, y, split, ks=(1, 2, 3)):
    """Pick k for similarity edges by validation macro-F1 only."""
    return max(ks, key=lambda k: train_gcn(NF, y, split, 'similarity', k)[2]['val_macro_f1'])


if __name__ == '__main__':
    # self-check: chain has 7 edges, union >= each part, normalised adjacency is symmetric
    rng = np.random.default_rng(0); NF = rng.normal(size=(3, 8, 19))
    (At, et), (As, es), (Au, eu) = (build_graphs(NF, m) for m in ('temporal', 'similarity', 'union'))
    assert At.shape == (3, 8, 8) and (et == 7).all() and (eu >= np.maximum(et, es)).all()
    assert np.allclose(Au, Au.transpose(0, 2, 1))
    print('task2 self-check ok')
