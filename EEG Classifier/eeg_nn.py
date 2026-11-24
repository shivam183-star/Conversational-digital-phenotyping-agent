"""
Neural-network (EEGNet-style CNN) classifier for the Mumtaz MDD vs Healthy EEG dataset.

Setup:
  1. Save your existing script as  eeg_features.py  in the same folder as this file
     (this file imports preprocess_and_clean_raw, DATA_DIR and SCALP_CHANNELS from it).
  2. pip install torch
  3. python eeg_nn.py

What it does:
  - Re-uses your preprocessing (filter -> ICA -> average ref -> 4 s epochs), then
    resamples to 128 Hz and z-scores every channel per recording (removes subject-level
    amplitude offsets, which really helps cross-subject generalisation).
  - Caches the preprocessed epochs to disk so ICA runs only once.
  - Trains a compact EEGNet-style CNN with augmentation, label smoothing, OneCycle LR.
  - Uses StratifiedGroupKFold: no subject ever appears in both train and test.
  - Ensembles several seeds and reports BOTH epoch-level and subject-level metrics
    (subject-level = average the epoch probabilities of each subject).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from training import preprocess_and_clean_raw, DATA_DIR, SCALP_CHANNELS

# ------------------------- config -------------------------
CACHE = Path("eeg_epochs_128hz.npz")
SFREQ = 128
N_FOLDS = 5
SEEDS = [0, 1, 2]          # ensemble size per fold
EPOCHS = 40                # fixed schedule, no checkpoint picking on the test fold
BATCH = 64
CROP = 384                 # random 3 s crop for training (full 4 s used at test time)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- data -------------------------
def build_or_load_cache():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["groups"]

    X_list, y_list, g_list = [], [], []
    for f in sorted(Path(DATA_DIR).glob("*.edf")):
        label = 1 if "MDD" in f.name.upper() else 0
        tokens = f.name.split()
        subject_id = f"{tokens[0]}_{tokens[1]}" if len(tokens) >= 2 else f.stem
        try:
            ep = preprocess_and_clean_raw(f, SCALP_CHANNELS)
            if len(ep) < 5:
                print(f"Skipping {f.name}: {len(ep)} epochs")
                continue
            ep.resample(SFREQ, verbose=False)
            data = ep.get_data() * 1e6                        # volts -> microvolts
            # per-recording, per-channel standardisation
            mu = data.mean(axis=(0, 2), keepdims=True)
            sd = data.std(axis=(0, 2), keepdims=True) + 1e-6
            data = ((data - mu) / sd).astype(np.float32)
            X_list.append(data)
            y_list += [label] * len(data)
            g_list += [subject_id] * len(data)
            print(f"{f.name}: {len(data)} epochs  ({subject_id}, label {label})")
        except Exception as e:
            print(f"Error on {f.name}: {e}")

    X = np.concatenate(X_list)
    y = np.array(y_list)
    groups = np.array(g_list)
    np.savez_compressed(CACHE, X=X, y=y, groups=groups)
    return X, y, groups


# ------------------------- model -------------------------
class EEGNet(nn.Module):
    """EEGNet-v4 style; global average pooling so it accepts any time length."""
    def __init__(self, n_ch=19, F1=8, D=2, F2=16, k1=64, drop=0.5, n_classes=2):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),   # spatial filters
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        self.head = nn.Linear(F2, n_classes)

    def forward(self, x):                   # x: (B, C, T)
        x = x.unsqueeze(1)                  # (B, 1, C, T)
        x = self.block2(self.block1(x))     # (B, F2, 1, T')
        x = x.mean(dim=(2, 3))              # global average pool
        return self.head(x)


# ------------------------- training utils -------------------------
def augment(xb):
    B, C, T = xb.shape
    start = np.random.randint(0, T - CROP + 1)
    xb = xb[:, :, start:start + CROP]
    xb = xb * torch.empty(B, 1, 1, device=xb.device).uniform_(0.8, 1.2)       # amplitude scaling
    xb = xb + 0.05 * torch.randn_like(xb)                                       # gaussian noise
    keep = (torch.rand(B, C, 1, device=xb.device) > 0.1).float()                # channel dropout
    return xb * keep


def predict_proba(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(DEVICE)
            out.append(F.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_one(Xtr, ytr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = EEGNet(n_ch=Xtr.shape[1]).to(DEVICE)

    counts = np.bincount(ytr, minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * counts), device=DEVICE)               # class balance
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)

    steps = int(np.ceil(len(Xtr) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * steps)

    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr).long()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            xb = augment(Xt[idx].to(DEVICE))
            yb = yt[idx].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
    return model


def subject_level(probs, y, groups):
    gs = np.unique(groups)
    p = np.array([probs[groups == g].mean() for g in gs])
    t = np.array([y[groups == g][0] for g in gs])
    return p, t


# ------------------------- main -------------------------
if __name__ == "__main__":
    X, y, groups = build_or_load_cache()
    print(f"\nX: {X.shape}  epochs: {len(y)}  subjects: {len(np.unique(groups))}  device: {DEVICE}")

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    ep_acc, ep_auc, sb_acc, sb_f1, sb_auc = [], [], [], [], []

    for fold, (tr, va) in enumerate(skf.split(X, y, groups)):
        probs = np.zeros(len(va))
        for s in SEEDS:
            model = train_one(X[tr], y[tr], seed=s)
            probs += predict_proba(model, X[va]) / len(SEEDS)

        pred = (probs > 0.5).astype(int)
        ea, eu = accuracy_score(y[va], pred), roc_auc_score(y[va], probs)

        sp, st = subject_level(probs, y[va], groups[va])
        sa = accuracy_score(st, sp > 0.5)
        sf = f1_score(st, sp > 0.5)
        su = roc_auc_score(st, sp) if len(np.unique(st)) > 1 else float("nan")

        ep_acc.append(ea); ep_auc.append(eu); sb_acc.append(sa); sb_f1.append(sf); sb_auc.append(su)
        print(f"Fold {fold + 1}: epoch acc {ea*100:.2f}% AUC {eu:.3f} | "
              f"subject acc {sa*100:.2f}% F1 {sf:.3f} AUC {su:.3f}  "
              f"(test subjects: {len(st)})")

    print("\n--- EEGNet ensemble, subject-wise 5-fold ---")
    print(f"Epoch-level   acc: {np.mean(ep_acc)*100:.2f}% ± {np.std(ep_acc)*100:.2f}% | AUC {np.mean(ep_auc):.3f}")
    print(f"Subject-level acc: {np.mean(sb_acc)*100:.2f}% ± {np.std(sb_acc)*100:.2f}% | "
          f"F1 {np.mean(sb_f1):.3f} | AUC {np.nanmean(sb_auc):.3f}")
