# Authors: Coumarane Tirou <c.tirou@hotmail.com>
# License: BSD (3-clause)

import os
import os.path as op
import mne
from base import *
from config import *
from mne import compute_covariance, compute_rank
from mne.beamformer import make_lcmv, apply_lcmv
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

# params
subjects = SUBJS15
lock = 'stim'
analysis = 'RSA'
data_path = DATA_DIR / 'for_rsa'
subjects_dir = FREESURFER_DIR

verbose = 'error'
overwrite = False
is_cluster = os.getenv("SLURM_ARRAY_TASK_ID") is not None

# pick_ori = 'vector'
# weight_norm = "unit-noise-gain-invariant"

pick_ori = 'max-power'
weight_norm = "unit-noise-gain"

subject = subjects[0]

res_path = ensured(RESULTS_DIR)

epochs = []
for epoch_num in range(5):
    epoch_fname = op.join(data_path, "epochs", f"{subject}-{epoch_num}-epo.fif")
    epoch = mne.read_epochs(epoch_fname, verbose=verbose, preload=True)
    epochs.append(epoch)

for epo in epochs:
    epo.info['dev_head_t'] = epochs[1].info['dev_head_t']
epochs = mne.concatenate_epochs(epochs)

# read forward solution
fwd_fname = RESULTS_DIR / "fwd" / "for_rsa" / f"{subject}-all-fwd.fif" # this fwd was not generated on the rdm_bsling data
fwd = mne.read_forward_solution(fwd_fname, verbose=verbose)

evoked = epochs.average()

plt.rcParams.update({'font.size': 14, 'font.family': 'serif', 'font.serif': 'Arial'})

fig, ax = plt.subplots(1, 1, figsize=(12, 4), layout='tight')
evoked.plot(axes=ax)
# remove the "N_ave" annotation
for text in list(ax.texts):
    text.remove()
# Remove spines and add grid
ax.set_axisbelow(True)
for key in ("top", "right"):
    ax.spines[key].set(visible=False)
# Tweak the ticks and limits
ax.set(
    yticks=np.arange(-200, 201, 100), xticks=np.arange(-0.1, 0.6, 0.1)
)
ax.set(ylim=[-225, 225], xlim=[-0.2, 0.6])
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
plt.savefig(res_path / f"{subject}-evoked.pdf", dpi=300, transparent=True)

noise_cov = compute_covariance(epochs, tmin=-0.2, tmax=0, method="empirical", rank="info", verbose=verbose)
data_cov = compute_covariance(epochs, method="empirical", rank="info", verbose=verbose)
rank = compute_rank(data_cov, info=epochs.info, rank=None, tol_kind='relative', verbose=verbose)
filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov,
                    pick_ori=pick_ori, weight_norm=weight_norm,
                    rank=rank, reduce_rank=True, verbose=verbose)
stc = apply_lcmv(evoked, filters=filters, verbose=verbose)

if stc.subject != 'fsaverage':
    morph = mne.compute_source_morph(stc, subject_from=stc.subject,
                                      subject_to='fsaverage',
                                      subjects_dir=subjects_dir)
    stc = morph.apply(stc)
stc.save(res_path / f"{subject}-morphed-all-stc.fif", overwrite=True)

# clim = dict(kind="value", lims=[4, 8, 12])
# clim = dict(kind='percent', lims=[80, 83, 90])
clim = dict(kind='percent', lims=[10, 30, 70])

brain = stc.plot(
    subject=stc.subject, hemi='lh', surface='pial',
    subjects_dir=subjects_dir, size=(2400, 1800), background='white',
    cortex='low_contrast', alpha=0.6,
    colormap='hot',
    # clim=clim,
    time_viewer=False, show_traces=False, colorbar=False,
)

brain.show_view("lateral")
plotter = brain.plotter

n_times = stc.data.shape[1]
hold = 5   # render each time point this many frames -> 4x slower

frames = []
for t in range(n_times):
    brain.set_time_point(t)
    plotter.render()
    img = plotter.screenshot(return_img=True, scale=2)
    for _ in range(hold):
        frames.append(img)

imageio.mimsave(
    res_path / 'brain_activations.mp4', frames, fps=60,
    codec='libx264', quality=9, macro_block_size=None,
    ffmpeg_params=['-crf', '16', '-preset', 'slow', '-pix_fmt', 'yuv420p'],
)
brain.close()
