import os
import re
import numpy as np
import pandas as pd
from astropy import units as u, constants as c

_SPECTRA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "diamondback_spectra")

def load_diamondback_spectrum(teff, grav, fsed, metallicity, co):
    """Load a Diamondback spectrum from data/diamondback_spectra.

    Parameters
    ----------
    teff : int
        Effective temperature in K (e.g. 1000).
    grav : int
        Surface gravity in m/s^2 (e.g. 316).
    fsed : int or str
        Sedimentation efficiency as an integer (e.g. 2) or "nc" for no-cloud.
    metallicity : float
        Log metallicity (e.g. 0.0, 0.5, -0.5).
    co : float
        C/O ratio (e.g. 1.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``wavelength`` (micron) and ``flux`` (W/m2/m),
        sorted by ascending wavelength.
    """
    if metallicity > 0:
        m_str = f"+{metallicity:.1f}"
    elif metallicity == 0.0:
        m_str = "0.0"
    else:
        m_str = f"{metallicity:.1f}"

    f_str = str(fsed)
    filename = f"t{teff}g{grav}{f_str}_m{m_str}_co{co:.1f}.spec"
    filepath = os.path.join(_SPECTRA_DIR, filename)

    with open(filepath, "r") as fh:
        content = fh.read()

    # The header spans 3 lines; line 4 ends with ']' followed by the first data
    # point on the same line.  Split on the closing bracket of the molecule list
    # to isolate everything that contains data.
    data_text = content.split("]", 1)[1]

    numbers = [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", data_text)]
    wavelength = np.array(numbers[0::2])
    flux = np.array(numbers[1::2])

    df = pd.DataFrame({"wavelength": wavelength, "flux": flux})
    return df.sort_values("wavelength").reset_index(drop=True)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
from joblib import Parallel, delayed

SIGMA_SB = float(c.sigma_sb / (u.W / u.m**2 / u.K**4))

def spectrum_to_teff(teff_in, grav, fsed, metallicity=0.0, co=1.0):
    """Return the Stefan-Boltzmann Teff from integrating a Diamondback spectrum."""
    df = load_diamondback_spectrum(teff_in, grav, fsed, metallicity, co)
    wl = np.array(df["wavelength"]) * 1e-6   # m
    fl = np.array(df["flux"])                 # W/m^2/m
    dwl = np.diff(wl)
    fl_mid = (fl[1:] + fl[:-1]) / 2
    integral = np.sum(fl_mid * dwl)
    return (integral / SIGMA_SB) ** 0.25


TEFFS  = list(range(900, 2500, 100))
GRAVS  = [31, 100, 316, 1000, 3160]
FSEDS  = ["nc", "f1", "f2", "f3", "f4", "f8"]

# ── compute ──────────────────────────────────────────────────────────────────
def _compute_one(teff_in, grav, fsed):
    try:
        return (teff_in, grav, fsed, spectrum_to_teff(teff_in, grav, fsed))
    except FileNotFoundError:
        return None

all_combos = list(itertools.product(TEFFS, GRAVS, FSEDS))
rows = Parallel(n_jobs=-1, verbose=5)(
    delayed(_compute_one)(t, g, f) for t, g, f in all_combos)

results = {(g, f): ([], []) for g in GRAVS for f in FSEDS}
for row in rows:
    teff_in, grav, fsed, teff_out = row
    results[(grav, fsed)][0].append(teff_in)
    results[(grav, fsed)][1].append(teff_out)
# sort each series by teff_in
results = {
    k: (np.array(tin := sorted(v[0])),
        np.array([tout for _, tout in sorted(zip(v[0], v[1]))]))
    for k, v in results.items()
}

# ── plot ──────────────────────────────────────────────────────────────────────
NCOLS = 3
NROWS = 2
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(14, 9), sharex=True, sharey=True)
axes = axes.flatten()

colors = cm.winter(np.linspace(0.1, 0.9, len(GRAVS)))
logg_labels = {31: "log g=3.5", 100: "log g=4.0", 316: "log g=4.5",
               1000: "log g=5.0", 3160: "log g=5.5"}

panel_titles = {"nc": "no cloud", "f1": "fsed=1", "f2": "fsed=2",
                "f3": "fsed=3", "f4": "fsed=4", "f8": "fsed=8"}

for ax, fsed in zip(axes, FSEDS):
    teff_range = np.array([min(TEFFS), max(TEFFS)])
    ax.plot(teff_range, teff_range, "k--", lw=1, alpha=0.5, label="1:1")
    for color, grav in zip(colors, GRAVS):
        tin, tout = results[(grav, fsed)]
        ax.plot(tin, tout, "o-", color=color, ms=3, lw=1.5,
                label=logg_labels[grav])
    ax.set_title(panel_titles[fsed], fontsize=11)
    ax.set_xlabel("Input $T_{\\mathrm{eff}}$ (K)")
    ax.set_ylabel("Output $T_{\\mathrm{eff}}$ (K)")
    ax.tick_params(labelbottom=True, labelleft=True)

# single legend outside panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(GRAVS) + 1,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Diamondback spectra: integrated $T_{\\mathrm{eff}}$ vs. input $T_{\\mathrm{eff}}$\n"
             r"(metallicity=0.0, C/O=1.0)", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 1])

outpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "diamondback_spectra_teff_integrals.png")
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved to {outpath}")
plt.show()