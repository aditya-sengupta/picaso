import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from calculate_t10 import t10
from matplotlib import cm

__refdata__ = os.environ["picaso_refdata"]
picaso_path = os.path.dirname(os.path.dirname(__refdata__))

bobcat_path = os.path.join(__refdata__, "sonora_grids", "bobcat")

count_available_overall, count_all_overall = 0, 0
for fsed in [1, 2, 3, 4, 8]:
    fsed_str = f"f{fsed}" if fsed != "nc" else "nc"
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    # log_gravs = [3.25, 3.5, 4, 4.5, 5, 5.5]
    # gravs = [17, 31, 100, 316, 1000, 3160]
    log_gravs = [3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5]
    gravs = [10, 17, 31, 56, 100, 177, 316, 562, 1000, 1778, 3160]
    semimajors = [0.02, 0.04, 0.13, 0.5, np.inf]
    teffs = np.arange(100, 2401, 50)
    cmap = cm.magma(np.linspace(0, 1, len(gravs) + 1))[:-1]

    for (k, semimajor) in enumerate(semimajors):
        all_teffs = []
        all_t10s = []
        count_available, count_all = 0, 0
        for j, (loggrav, grav, c) in enumerate(zip(log_gravs, gravs, cmap)):
            teffs_this = []
            t10s = []
            for teff in teffs:
                semimajor_str = (
                    "ns" if semimajor == np.inf else f"semimajor{semimajor:.2f}"
                )
                fname = f"data/unified/unified_tint{teff}_grav{grav}_{semimajor_str}_{fsed_str}.h5"
                count_all += 1
                count_all_overall += 1
                if os.path.exists(os.path.join(picaso_path, fname)):
                    try:
                        with h5py.File(os.path.join(picaso_path, fname)) as f:
                            if "pressure" in f:
                                count_available += 1
                                count_available_overall += 1
                                pressure, temperature = (
                                    np.array(f["pressure"]),
                                    np.array(f["temperature"]),
                                )
                                teffs_this.append(teff)
                                if "t10" in f.attrs:
                                    t10s.append(f.attrs["t10"])
                                else:
                                    t10s.append(t10(pressure, temperature))
                    except Exception:
                        continue
                else:
                    pass

            all_teffs.append(teffs_this)
            all_t10s.append(t10s)

            t = f"a = {semimajor} au" if semimajor < np.inf else "no star"
            curr_ax = axs[k // 3, k % 3]
            curr_ax.plot(teffs_this, t10s, color=c, label=f"log g = {loggrav}")
            curr_ax.invert_yaxis()
            axs[1, k % 3].set_xlabel("Tint (K)")
            curr_ax.set_ylabel("T10 (K)")
            curr_ax.set_xlim((np.min(teffs) - 10, np.max(teffs) + 10))
            curr_ax.set_ylim((0, 5000))
            curr_ax.set_title(f"{t}, {fsed_str}")
            if k % 3 > 0:
                curr_ax.yaxis.set_visible(False)
            if k // 3 == 0:
                curr_ax.xaxis.set_visible(False)

        print(f"{fsed = }, {semimajor = }: {count_available} / {count_all}")
        T10_table = 0.1 * np.ones((len(gravs), len(teffs)))
        for j, grav in enumerate(gravs):
            if len(all_teffs[j]) > 1:
                T10_table[j, :] = np.interp(teffs, all_teffs[j], all_t10s[j])

        np.savez(
            f"data/t10_tables/atm_semimajor{semimajor:.2f}_{fsed_str}.npz",
            logGravity=log_gravs,
            logTeff=np.log10(teffs),
            logT10=np.log10(T10_table),
        )

    axs[1,1].legend(fontsize="small", bbox_to_anchor=(1.2, 1.05))
    fig.delaxes(axs[1,2])
    figpath = os.path.join(picaso_path, f"figures/t10/t10_table_{fsed_str}.png")
    plt.savefig(figpath, dpi=600)
    print(figpath)
    plt.close(fig)

print(f"{count_available_overall} / {count_all_overall}")
