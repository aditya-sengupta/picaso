import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

picaso_path = "/Users/adityasengupta/picaso"
out_dir = os.path.join(picaso_path, "figures/userdefinedclouds_comparison")
os.makedirs(out_dir, exist_ok=True)

old = pkl.load(open(os.path.join(picaso_path, 'data/bd_fixed_2602/bd_cloudmodefixed_fsed2_teff200_grav3160_semimajor0.08_prev.pkl'), 'rb'))
new = pkl.load(open(os.path.join(picaso_path, 'data/bd_fixed_2602/bd_cloudmodefixed_fsed2_teff200_grav3160_semimajor0.08.pkl'), 'rb'))

print('=== Keys ===')
print('Old:', sorted(old.keys()))
print('New:', sorted(new.keys()))

print()
print('=== Final temperature ===')
print('Old T range:', np.min(old['temperature']), '-', np.max(old['temperature']))
print('New T range:', np.min(new['temperature']), '-', np.max(new['temperature']))
print('Max |dT|:', np.max(np.abs(old['temperature'] - new['temperature'])))
print('Old len(T):', len(old['temperature']))
print('New len(T):', len(new['temperature']))

nlev = len(old['temperature'])
print()
print('=== all_profiles ===')
print('nlev:', nlev)
print('Old all_profiles len:', len(old['all_profiles']))
print('New all_profiles len:', len(new['all_profiles']))

old_init = old['all_profiles'][:nlev]
new_init = new['all_profiles'][:nlev]
print('Old initial T range:', np.min(old_init), '-', np.max(old_init))
print('New initial T range:', np.min(new_init), '-', np.max(new_init))
print('Max |dT_init|:', np.max(np.abs(old_init - new_init)))

print()
print('=== Convergence ===')
print('Old converged:', old.get('converged', 'N/A'))
print('New converged:', new.get('converged', 'N/A'))
old_niters = len(old['all_profiles']) // nlev
new_niters = len(new['all_profiles']) // nlev
print('Old n_iters:', old_niters)
print('New n_iters:', new_niters)

# --- Plot comparison ---
pressure = old['pressure']

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left panel: initial guesses
ax = axes[0]
ax.semilogy(old_init, pressure, label="Old initial guess", lw=3, ls="--", color="C0", zorder=1)
ax.semilogy(new_init, pressure, label="New initial guess", lw=1.5, ls=":", color="C1", zorder=2)
ax.semilogy(old['temperature'], pressure, label="Old final", lw=2, color="C0")
ax.semilogy(new['temperature'], pressure, label="New final", lw=2, color="C1")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (bar)")
ax.invert_yaxis()
ax.legend(fontsize=8)
ax.set_title("T-P profiles: old vs new")

# Right panel: temperature difference
ax = axes[1]
ax.semilogx(pressure, old['temperature'] - new['temperature'])
ax.set_ylabel("Old T - New T (K)")
ax.set_xlabel("Pressure (bar)")
ax.axhline(0, ls=":", color="gray")
ax.set_title("Final temperature difference")

plt.suptitle("g=3160, Teff=200, semi_major=0.08, fixed clouds", fontsize=12)
plt.tight_layout()
fig_fname = os.path.join(out_dir, "compare_old_new_teff200_grav3160.png")
plt.savefig(fig_fname, dpi=150)
plt.close()
print(f"\nSaved {fig_fname}")
