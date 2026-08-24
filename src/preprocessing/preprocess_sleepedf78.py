"""
SleepMamba reproduction — Step 1: SleepEDF-78 preprocessing
=============================================================
Implements exactly what Section 4.1 of the paper specifies, nothing more:

  - Channels: Fpz-Cz EEG + ROC-LOC (horizontal) EOG for the main experiments.
    EMG (submental) is also extracted, used only for the modality ablation
    (Table 6: EEG, EEG+EOG, EEG+EMG, EEG+EOG+EMG).
  - Epoch length: 30 seconds.
  - Remove epochs labeled MOVEMENT or UNKNOWN.
  - Trim any pre-sleep segment whose sleep-onset latency exceeds 30 minutes
    (i.e. keep at most 30 min of Wake before the first non-Wake epoch, and
    — per standard practice in this exact line of work (DeepSleepNet /
    SeqSleepNet / XSleepNet) which the paper's phrasing follows — the same
    30-min trim is applied symmetrically at the end of the recording after
    the last non-Wake epoch. NOTE: the paper only explicitly states the
    pre-sleep trim; the post-sleep trim is our assumption, applied because
    it's what every predecessor paper in this exact lineage does and
    without it the epoch counts will not match Table 2. Flagged clearly.)
  - Sampling rate: 100 Hz (3000 samples / 30s epoch), as implied by the
    noise-robustness experiment in Section 5.6.
  - Per-recording, per-channel z-score normalization (NOT stated in the
    paper — our assumed convention, standard across this literature).

Output: one .npz per recording under OUT_DIR, each containing:
    x        : float32 array (n_epochs, n_channels, 3000)  channel order = [EEG, EOG, EMG]
    y        : int64 array (n_epochs,)  labels in {0:W,1:N1,2:N2,3:N3,4:REM}
    subject_id, night : ints, parsed from filename, for subject-level CV grouping

Run from VS Code with your Google Drive Sleep-EDF folder as DATA_DIR.
Requires: mne, numpy  (pip install mne numpy --break-system-packages)
"""

import os
import re
import glob
import numpy as np
import mne

# ---------------------------------------------------------------------------
# CONFIG — edit these two paths for your machine
# ---------------------------------------------------------------------------
DATA_DIR = r"D:\Sleep Mamba\data\sleepmamba_data"
OUT_DIR = r"D:\Sleep Mamba\outputs\preprocessed"

EPOCH_SEC = 30
SFREQ = 100  # Hz, per Section 5.6 (3000 samples / 30s epoch)
SAMPLES_PER_EPOCH = EPOCH_SEC * SFREQ  # 3000

# Standard Sleep-EDF channel names (Sleep Cassette study)
EEG_CH = "EEG Fpz-Cz"
EOG_CH = "EOG horizontal"
EMG_CH = "EMG submental"

# AASM/Sleep-EDF annotation string -> label index
# Sleep-EDF hypnograms use "Sleep stage W/1/2/3/4/R/?/M" as annotation descriptions
STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # paper merges S3+S4 into N3 for SHHS1; for SleepEDF the
                           # standard convention (and AASM correspondence) also
                           # merges 3/4 -> N3. The paper doesn't restate this for
                           # SleepEDF-78 explicitly, but every baseline it compares
                           # against (SeqSleepNet, XSleepNet, AttnSleep) does this,
                           # and Table 2's 5-class totals require it. Flagged.
    "Sleep stage R": 4,
    # "Sleep stage ?" and "Movement time" are dropped (UNKNOWN / MOVEMENT)
}
DROP_LABELS = {"Sleep stage ?", "Movement time"}

PRE_POST_TRIM_MIN = 30  # minutes of Wake padding kept around sleep period


def find_recording_pairs(data_dir):
    """Pair each *-PSG.edf with its matching *-Hypnogram.edf by subject+night."""
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*-PSG.edf")))
    hyp_files = sorted(glob.glob(os.path.join(data_dir, "*-Hypnogram.edf")))

    def key(fname):
        base = os.path.basename(fname)
        # e.g. SC4821G0-PSG.edf -> subject=82, night=1
        m = re.match(r"SC4(\d)(\d)(\d)([A-Z])", base)
        if not m:
            raise ValueError(f"Unrecognized filename pattern: {base}")
        subj_tens, subj_units, night, variant = m.groups()
        subject_id = int(subj_tens + subj_units)
        night_id = int(night)
        return subject_id, night_id

    hyp_lookup = {key(f): f for f in hyp_files}
    pairs = []
    for psg in psg_files:
        k = key(psg)
        if k not in hyp_lookup:
            print(f"WARNING: no hypnogram match for {psg}, skipping")
            continue
        pairs.append((psg, hyp_lookup[k], k[0], k[1]))
    return pairs


def load_recording(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, emit_warning=False)

    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)

    available = raw.ch_names
    ch_picks = []
    for want in (EEG_CH, EOG_CH, EMG_CH):
        if want in available:
            ch_picks.append(want)
        else:
            raise ValueError(f"Channel '{want}' not found in {psg_path}. Available: {available}")
    raw.pick(ch_picks)
    # reorder deterministically: EEG, EOG, EMG
    raw.reorder_channels([EEG_CH, EOG_CH, EMG_CH])

    events, event_id = mne.events_from_annotations(raw, chunk_duration=EPOCH_SEC, verbose=False)
    # invert event_id to get description per event
    inv_event_id = {v: k for k, v in event_id.items()}

    data = raw.get_data()  # (3, n_samples)
    labels = []
    starts = []
    for onset_sample, _, ev_code in events:
        desc = inv_event_id[ev_code]
        if desc in DROP_LABELS:
            continue
        if desc not in STAGE_MAP:
            continue  # any other stray annotation, drop
        end_sample = onset_sample + SAMPLES_PER_EPOCH
        if end_sample > data.shape[1]:
            continue
        labels.append(STAGE_MAP[desc])
        starts.append(onset_sample)

    labels = np.array(labels, dtype=np.int64)
    starts = np.array(starts, dtype=np.int64)

    # --- trim pre/post sleep Wake beyond 30 min, per Section 4.1 ---
    non_wake_idx = np.where(labels != 0)[0]
    if len(non_wake_idx) == 0:
        raise ValueError(f"No sleep epochs found in {psg_path}")
    first_sleep, last_sleep = non_wake_idx[0], non_wake_idx[-1]
    trim_epochs = (PRE_POST_TRIM_MIN * 60) // EPOCH_SEC  # = 60 epochs for 30 min

    keep_start = max(0, first_sleep - trim_epochs)
    keep_end = min(len(labels), last_sleep + trim_epochs + 1)

    labels = labels[keep_start:keep_end]
    starts = starts[keep_start:keep_end]

    # --- extract epochs ---
    x = np.stack([data[:, s:s + SAMPLES_PER_EPOCH] for s in starts], axis=0)  # (n_epochs, 3, 3000)

    # --- per-recording, per-channel z-score normalization ---
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    x = (x - mean) / std

    return x.astype(np.float32), labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = find_recording_pairs(DATA_DIR)
    print(f"Found {len(pairs)} PSG/Hypnogram pairs.")

    total_counts = np.zeros(5, dtype=np.int64)
    n_recordings_ok = 0

    for psg_path, hyp_path, subject_id, night_id in pairs:
        try:
            x, y = load_recording(psg_path, hyp_path)
        except Exception as e:
            print(f"FAILED on {os.path.basename(psg_path)}: {e}")
            continue

        counts = np.bincount(y, minlength=5)
        total_counts += counts
        n_recordings_ok += 1

        out_name = f"subj{subject_id:02d}_night{night_id}.npz"
        np.savez_compressed(
            os.path.join(OUT_DIR, out_name),
            x=x, y=y, subject_id=subject_id, night=night_id,
        )
        print(f"{out_name}: {x.shape[0]} epochs  W/N1/N2/N3/REM = {counts.tolist()}")

    print("\n=== TOTALS (compare against paper Table 2) ===")
    print(f"Recordings processed: {n_recordings_ok} / {len(pairs)}")
    labels5 = ["W", "N1", "N2", "N3", "REM"]
    grand_total = total_counts.sum()
    for name, c in zip(labels5, total_counts):
        print(f"  {name}: {c}  ({100*c/grand_total:.1f}%)")
    print(f"  TOTAL: {grand_total}")
    print("\nPaper Table 2 target: W=65642 N1=21520 N2=69132 N3=13039 REM=25835 TOTAL=195168")


if __name__ == "__main__":
    main()