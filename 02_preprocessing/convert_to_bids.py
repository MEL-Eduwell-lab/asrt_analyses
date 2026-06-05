# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)
#
# Convert the raw 4D/BTi (Magnes) MEG recordings stored under
# /Volumes/Ultra_Touch/asrt/raws into an anonymized BIDS dataset.
#
# Layout of the source data (one folder per subject, one folder per run):
#   sub<NN>/meg_data/<RUN>/results/c,rfDC_EEG   <- PDF (data) read by the pipeline
#   sub<NN>/meg_data/<RUN>/results/config       <- 4D config file
#   sub<NN>/meg_data/<RUN>/results/hs_file      <- head-shape file (not always present)
#
# Runs (see config.EPOCHS):
#   2_PRACTICE, 3_EPOCH_1, 4_EPOCH_2, 5_EPOCH_3, 6_EPOCH_4
#   (+ resting-state runs that exist for sub11 only)
#
# Run interactively (Jupyter) or as a module:
#   python -m 02_preprocessing.convert_to_bids

import os
from pathlib import Path

import mne
from mne_bids import (
    BIDSPath,
    write_raw_bids,
    make_dataset_description,
    get_anonymization_daysback,
    print_dir_tree,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
input_dir = Path("/Volumes/Ultra_Touch/asrt/raws")
bids_root = Path("/Volumes/Ultra_Touch/asrt/bids_data")

LINE_FREQ = 50.0          # CRNL / MEG-Lyon recordings: 50 Hz mains
POWER_LINE = "n/a"
OVERWRITE = True

# Flat / noisy MEG sensors common to all subjects (see save_epochs.py).
# Marked as bad rather than dropped, so the shared dataset stays complete.
KNOWN_BADS = ["MEG 059", "MEG 173", "MEG 028"]

# Map each acquisition folder to a BIDS (task, run). run=None -> omitted.
SESSION_MAP = {
    "2_PRACTICE":        ("practice", None),
    "3_EPOCH_1":         ("asrt", 1),
    "4_EPOCH_2":         ("asrt", 2),
    "5_EPOCH_3":         ("asrt", 3),
    "6_EPOCH_4":         ("asrt", 4),
    "1_RESTING_STATE_1": ("rest", 1),
    "7_RESTING_STATE_2": ("rest", 2),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_bti_files(session_dir):
    """Locate the PDF, config and (optional) head-shape file for a run.

    Returns (pdf, config, head_shape) as strings, or None if no PDF is found.
    head_shape is None when the hs_file is absent.
    """
    results = session_dir / "results"
    # Most runs keep the EEG-merged PDF under results/; a few raw runs
    # (e.g. sub11/RESTARTED_RUN) keep the original c,rfDC at the run root.
    candidates = [
        (results / "c,rfDC_EEG", results),
        (results / "c,rfDC", results),
        (session_dir / "c,rfDC_EEG", session_dir),
        (session_dir / "c,rfDC", session_dir),
    ]
    for pdf, base in candidates:
        if pdf.exists():
            config = base / "config"
            if not config.exists():
                continue
            hs = base / "hs_file"
            head_shape = str(hs) if hs.exists() else None
            return str(pdf), str(config), head_shape
    return None


def read_run(pdf, config, head_shape):
    """Read a 4D/BTi run and tidy up channel metadata for sharing."""
    raw = mne.io.read_raw_bti(
        pdf,
        config_fname=config,
        head_shape_fname=head_shape,
        preload=True,
        verbose="ERROR",
    )

    # Assign the known auxiliary channel types (matches save_epochs.py).
    type_map, rename = {}, {}
    if "EEG 001" in raw.ch_names:
        type_map["EEG 001"] = "ecg"
        rename["EEG 001"] = "ECG 001"
    for ch in ("VEOG", "HEOG"):
        if ch in raw.ch_names:
            type_map[ch] = "eog"
    if "UTL 001" in raw.ch_names:
        type_map["UTL 001"] = "misc"
        rename["UTL 001"] = "MISC 001"
    if type_map:
        raw.set_channel_types(type_map)
    if rename:
        raw.rename_channels(rename)

    # Mark the flat/noisy sensors as bad instead of dropping them.
    raw.info["bads"] = [c for c in KNOWN_BADS if c in raw.ch_names]
    raw.info["line_freq"] = LINE_FREQ
    return raw


def collect_runs():
    """Walk the source tree and yield (subject, task, run, paths) tuples."""
    for subject_dir in sorted(input_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("sub"):
            continue
        subject = subject_dir.name[-2:]  # 'sub01' -> '01'
        meg_dir = subject_dir / "meg_data"
        if not meg_dir.is_dir():
            continue
        for session_dir in sorted(meg_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            mapping = SESSION_MAP.get(session_dir.name)
            if mapping is None:
                print(f"  ! skipping unmapped run: {session_dir}")
                continue
            task, run = mapping
            paths = find_bti_files(session_dir)
            if paths is None:
                print(f"  ! no BTi PDF found in: {session_dir}")
                continue
            yield subject, task, run, paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    bids_root.mkdir(parents=True, exist_ok=True)
    runs = list(collect_runs())
    if not runs:
        raise RuntimeError(f"No convertible runs found under {input_dir}")

    # A single, consistent date shift for the whole dataset preserves the
    # relative timing between runs/subjects while anonymizing acquisition dates.
    print("Computing anonymization date shift from acquisition headers...")
    headers = [
        mne.io.read_raw_bti(pdf, config_fname=cfg, head_shape_fname=hs,
                            preload=False, verbose="ERROR")
        for _, _, _, (pdf, cfg, hs) in runs
    ]
    daysback_min, _ = get_anonymization_daysback(headers)
    del headers
    print(f"  -> daysback = {daysback_min}")

    for subject, task, run, (pdf, cfg, hs) in runs:
        print(f"\n[sub-{subject}] task-{task}"
            + (f" run-{run:02d}" if run is not None else ""))
        raw = read_run(pdf, cfg, hs)

        bids_path = BIDSPath(
            subject=subject,
            task=task,
            run=f"{run:02d}" if run is not None else None,
            root=bids_root,
            datatype="meg",
        )
        write_raw_bids(
            raw,
            bids_path,
            anonymize=dict(daysback=daysback_min, keep_his=False),
            format="FIF",
            allow_preload=True,
            overwrite=OVERWRITE,
            verbose="ERROR",
        )
        del raw

    # Top-level dataset metadata.
    make_dataset_description(
        path=bids_root,
        name="Learning regularities in noise (ASRT MEG)",
        dataset_type="raw",
        authors=["Coumarane Tirou"],
        references_and_links=["https://doi.org/10.1101/2025.08.18.670891"],
        overwrite=True,
    )

    print("\nBIDS dataset created at", bids_root)
    print_dir_tree(bids_root, max_depth=4)


if __name__ == "__main__":
    main()
