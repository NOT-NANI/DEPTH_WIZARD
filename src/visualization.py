"""Diagnostic figure generation for the single GAMUS sample."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save_diagnostics(rgb, raw, gt, aligned, valid, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    err = np.where(valid, aligned - gt, np.nan)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb); axes[0, 0].set_title("GAMUS RGB")
    im=axes[0, 1].imshow(raw,cmap="magma"); axes[0, 1].set_title("Raw prediction (relative units)"); fig.colorbar(im,ax=axes[0,1],shrink=.75)
    im=axes[0, 2].imshow(np.where(valid,gt,np.nan),cmap="terrain"); axes[0, 2].set_title("Ground-truth AGL (m)"); fig.colorbar(im,ax=axes[0,2],shrink=.75)
    im=axes[1, 0].imshow(np.where(valid,aligned,np.nan),cmap="terrain",vmin=np.nanpercentile(gt[valid],2),vmax=np.nanpercentile(gt[valid],98)); axes[1,0].set_title("Scale/shift aligned prediction (m)"); fig.colorbar(im,ax=axes[1,0],shrink=.75)
    lim=float(np.nanpercentile(np.abs(err),98))
    im=axes[1,1].imshow(err,cmap="RdBu_r",vmin=-lim,vmax=lim); axes[1,1].set_title("Aligned error: prediction − GT (m)"); fig.colorbar(im,ax=axes[1,1],shrink=.75)
    axes[1,2].hist(err[valid],bins=80,color="#54708c"); axes[1,2].set_title("Valid-pixel error distribution"); axes[1,2].set_xlabel("Error (m)"); axes[1,2].set_ylabel("Pixels")
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    figure=output/"gamus_baseline_diagnostics.png"; fig.savefig(figure,dpi=150); plt.close(fig)
    error=output/"gamus_error_map.png"; plt.imsave(error,err,cmap="RdBu_r",vmin=-lim,vmax=lim)
    return figure,error
