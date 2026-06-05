import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from base import get_sequence
from config import SUBJS15, HOME

subjects = SUBJS15
sub_nums = [int(sub[3:]) for sub in subjects]
epoch_nums = list(range(1, 5))

data_dir = Path("/Users/coum/Desktop/explicit_test")

# --- Build sequence lookup {sub_num: [p1, p2, p3, p4]} ---
sequences = {}
for sub, sub_num in zip(subjects, sub_nums):
    behav_dir = op.join(HOME / 'raw_behavs' / sub)
    sequences[sub_num] = get_sequence(behav_dir)


def compute_accuracy(responses, anchor, sequence):
    """
    Accuracy of 12 responses against the true cyclic ASRT sequence.

    The anchor (response_idx == 0) is the last key from the task; it fixes
    which position in the cycle the participant should continue from.
    If the anchor is not in the sequence (data anomaly), fall back to the
    best alignment over all 4 phases.

    Returns accuracy as a percentage (0–100).
    """
    seq = list(sequence)
    n = len(responses)

    if anchor in seq:
        phase = (seq.index(anchor) + 1) % 4
        expected = [seq[(phase + i) % 4] for i in range(n)]
        correct = sum(r == e for r, e in zip(responses, expected))
    else:
        # anchor not in sequence: pick best-fitting phase
        correct = max(
            sum(r == seq[(start + i) % 4] for i, r in enumerate(responses))
            for start in range(4)
        )

    return correct / n * 100


# --- Score every subject × epoch × block ---
rows = []
for sub_num in tqdm(sub_nums, desc="subjects"):
    seq = sequences[sub_num]
    for epoch in epoch_nums:
        fpath = data_dir / f"{sub_num}_seq_report_Epoch_{epoch}.txt"
        df = pd.read_csv(fpath, sep="\t", header=None,
                         names=["subject", "block_within_epoch", "response_idx", "key"])

        for block in sorted(df["block_within_epoch"].unique()):
            block_df = df[df["block_within_epoch"] == block]
            anchor = block_df.loc[block_df["response_idx"] == 0, "key"].iloc[0]
            responses = (block_df[block_df["response_idx"] > 0]
                         .sort_values("response_idx")["key"]
                         .tolist())

            acc = compute_accuracy(responses, anchor, seq)
            rows.append({
                "subject": sub_num,
                "block_within_epoch": block,
                "accuracy": acc,
            })

results = pd.DataFrame(rows)
print(results)

# Save
out_path = data_dir / "explicit_scores.csv"
results.to_csv(out_path, index=False)

# --- Plot: mean accuracy ± SEM across subjects per block_within_epoch ---
stats = results.groupby("block_within_epoch")["accuracy"].agg(
    mean="mean", sem=lambda x: x.std() / np.sqrt(len(x))
).reset_index()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(stats["block_within_epoch"], stats["mean"], marker="o", color="#0072B2")
ax.fill_between(
    stats["block_within_epoch"],
    stats["mean"] - stats["sem"],
    stats["mean"] + stats["sem"],
    alpha=0.25, color="#0072B2"
)
ax.axhline(25, linestyle="--", color="gray", linewidth=0.8, label="chance (25%)")
ax.set_xlabel("Block within epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Explicit sequence knowledge")
ax.set_xticks(stats["block_within_epoch"])
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.show()
