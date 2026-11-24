import os
import re
from pathlib import Path
import mne
import numpy as np
from mne.preprocessing import ICA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DATA_DIR = Path(r"C:\EEG Classification\data\raw")  # Adjust to your folder path

SCALP_CHANNELS = [
    "EEG Fp1-LE", "EEG F3-LE", "EEG C3-LE", "EEG P3-LE", "EEG O1-LE",
    "EEG F7-LE",  "EEG T3-LE", "EEG T5-LE", "EEG Fz-LE", "EEG Fp2-LE",
    "EEG F4-LE",  "EEG C4-LE", "EEG P4-LE", "EEG O2-LE", "EEG F8-LE",
    "EEG T4-LE",  "EEG T6-LE", "EEG Cz-LE", "EEG Pz-LE"
]

def preprocess_and_clean_raw(file_path, scalp_channels, l_freq=1.0, h_freq=45.0, notch_freq=50.0):
    """
    Loads EDF, cleans channel labels, applies filters, rejects/interpolates bad channels,
    runs ICA to subtract ocular/cardiac artifacts, re-references to average,
    and returns 4.0s epochs with an artifact threshold of 250 µV.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    rename_dict = {ch: ch.replace("EEG ", "").replace("-LE", "").strip() for ch in raw.ch_names}
    raw.rename_channels(rename_dict)
    clean_scalp_ch = [ch.replace("EEG ", "").replace("-LE", "").strip() for ch in scalp_channels]
    raw.pick(clean_scalp_ch)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    
    data = raw.get_data()
    ch_ptps = np.ptp(data, axis=1)
    median_ptp = np.median(ch_ptps)
    std_ptp = np.std(ch_ptps)
    
    bad_channels = [
        raw.ch_names[i] for i, ptp in enumerate(ch_ptps) 
        if ptp > (median_ptp + 5 * std_ptp) or ptp < 1e-6
    ]
    raw.info['bads'] = bad_channels

    # Independent Component Analysis (ICA)
    n_components = min(15, len(raw.ch_names) - len(bad_channels))
    ica = ICA(n_components=n_components, random_state=42, max_iter=800)
    ica.fit(raw, picks='eeg', reject_by_annotation=True, verbose=False)
    
    # Auto-detect ocular blink components using frontal sensors
    eog_indices = []
    for test_ch in ['Fp1', 'Fp2']:
        if test_ch in raw.ch_names and test_ch not in raw.info['bads']:
            scores = ica.score_sources(raw, target=test_ch, score_func='pearsonr', verbose=False)
            top_matches = np.where(np.abs(scores) > 0.40)[0].tolist()
            eog_indices.extend(top_matches)
            
    ica.exclude = list(set(eog_indices))[:2]
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned, verbose=False)

    if len(raw_cleaned.info['bads']) > 0:
        raw_cleaned.interpolate_bads(reset_bads=True, verbose=False)
    raw_cleaned.set_eeg_reference('average', projection=False, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw_cleaned, duration=4.0, overlap=1.0, preload=True, verbose=False
    )
    
    epochs.drop_bad(reject=dict(eeg=250e-6), verbose=False)
    return epochs

def extract_epoch_features(epochs):
    """
    Extracts relative band powers (delta, theta, alpha, beta, gamma)
    and Frontal Alpha Asymmetry (FAA = ln(F4) - ln(F3)).
    """
    spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=45, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    total_mask = (freqs >= 1) & (freqs <= 45)
    total_power = np.trapezoid(psds[:, :, total_mask], freqs[total_mask], axis=2)
    
    features = []
    for _, (low, high) in bands.items():
        b_mask = (freqs >= low) & (freqs < high)
        bp = np.trapezoid(psds[:, :, b_mask], freqs[b_mask], axis=2)
        rel_bp = bp / (total_power + 1e-12)
        features.append(rel_bp)
        
    X_psd = np.concatenate(features, axis=1)  # (n_epochs, n_channels * 5)
    
    # Calculate Frontal Alpha Asymmetry (F4 vs F3)
    ch_names = epochs.ch_names
    if 'F4' in ch_names and 'F3' in ch_names:
        f4_idx, f3_idx = ch_names.index('F4'), ch_names.index('F3')
        a_mask = (freqs >= 8) & (freqs < 13)
        alpha_f4 = np.trapezoid(psds[:, f4_idx, a_mask], freqs[a_mask], axis=1)
        alpha_f3 = np.trapezoid(psds[:, f3_idx, a_mask], freqs[a_mask], axis=1)
        faa = (np.log(alpha_f4 + 1e-12) - np.log(alpha_f3 + 1e-12))[:, None]
        X_psd = np.hstack([X_psd, faa])
        
    return X_psd

def build_dataset(data_dir, scalp_channels):
    """
    Iterates through all EDF files in data_dir, extracts features for ML,
    extracts raw epoch arrays for DL, and labels subjects to prevent data leakage.
    """
    X_list, y_list, group_list = [], [], []
    files = list(Path(data_dir).glob("*.edf"))
    print(f"Found {len(files)} EDF files in {data_dir}...")
    
    for f in files:
        label = 1 if "MDD" in f.name.upper() else 0
        
        tokens = f.name.split()
        subject_id = f"{tokens[0]}_{tokens[1]}" if len(tokens) >= 2 else f.stem
        
        try:
            epochs = preprocess_and_clean_raw(f, scalp_channels)
            if len(epochs) < 5:
                print(f"Skipping {f.name}: only {len(epochs)} valid epochs retained.")
                continue
            
            feats = extract_epoch_features(epochs)
            X_list.append(feats)
            
            y_list.extend([label] * len(epochs))
            group_list.extend([subject_id] * len(epochs))
            print(f"Processed {f.name} -> {len(epochs)} epochs (Subject: {subject_id}, Label: {label})")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(group_list)
    
    return X, y, groups

def train_classical_ml(X_ml, y, groups):
    print("RUNNING XGBOOST")
    
    gkf = GroupKFold(n_splits=5)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_ml, y, groups=groups)):
        X_train, X_val = X_ml[train_idx], X_ml[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        clf = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, 
            eval_metric='logloss', random_state=42
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)[:, 1]
        
        fold_acc = accuracy_score(y_val, preds)
        fold_f1 = f1_score(y_val, preds)
        fold_auc = roc_auc_score(y_val, probs)
        
        accs.append(fold_acc)
        f1s.append(fold_f1)
        aucs.append(fold_auc)
        print(f"Fold {fold + 1} -> Accuracy: {fold_acc*100:.2f}%, F1: {fold_f1:.3f}, AUC: {fold_auc:.3f}")

    print(f"\n--- XGBoost 5-Fold Summary ---")
    print(f"Mean Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"Mean F1-Score: {np.mean(f1s):.3f} | Mean ROC-AUC: {np.mean(aucs):.3f}")


if __name__ == "__main__":

    X, y, groups = build_dataset(DATA_DIR, SCALP_CHANNELS)
    print(f"\nExtracted Dataset Shapes:")
    print(f"ML Matrix (X): {X.shape}")
    print(f"Total Epochs: {len(y)} | Unique Subjects: {len(np.unique(groups))}")
    
    train_classical_ml(X, y, groups)