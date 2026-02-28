"""
Diagnosis of differences between old (a19a616) and new (fixedclouds) fixed-cloud behavior.

This script traces the logic in both versions and produces figures explaining
the three differences found.

Case: g=3160, Teff=200, semi_major=0.08, fsed=2, fixed clouds.

=== SUMMARY OF FINDINGS ===

There are two logic differences between the old and new code for cloudy="fixed":

DIFFERENCE 1: Cloud opacity ramp-in on the first profile() call (0.25x vs 1.0x)
  Old code (a19a616):
    - update_clouds("fixed") -> becomes "fixed_after_first", calls update_clouds_selfconsistent()
    - update_clouds_selfconsistent runs virga on the initial T-P profile
    - It uses a 4-step weighted average: OPD_eff = 0.25*virga + 0.25*0 + 0.25*0 + 0.25*0 = 0.25*virga
    - bundle.clouds() is called with this 25% opacity
    - calculate_atm() then picks up this 25% cloud for the first iteration
    - All subsequent iterations in the first profile() call use this same 25% opacity
      (update_clouds sets full opacity via bundle.clouds(), but calculate_atm() is NOT called
       because refresh_needed = cloudy == "selfconsistent" is False for cloudy="fixed")
    - Starting from the second profile() call (STEP 2 of run_chemeq_climate_workflow),
      cloudy is now "fixed_after_first", and the pre-loop section calls calculate_atm()
      which picks up the full virga OPD

  New code (fixedclouds branch):
    - self.clouds(df=df_cld) is called before the workflow with the FULL user-provided cloud
    - update_clouds("fixed") is a no-op (returns immediately)
    - calculate_atm() picks up the full cloud opacity from the start
    - All profile() calls see 100% of the cloud from iteration 1

  Impact: The old code effectively "ramps in" the cloud over the first ~10 iterations by
  starting at 25%. The new code starts at 100% immediately. For optically thick clouds
  (max OPD ~5500 in this case), this is a massive difference in the first profile() call
  and can push the temperature structure to a very different convergence path.

DIFFERENCE 2: In-loop opacity refresh for fixed clouds
  Old code:
    - In the main loop, update_clouds("fixed_after_first") calls bundle.clouds() with
      OPD[:,:,0] (full virga result) every iteration
    - But calculate_atm() is NOT re-called (refresh_needed = cloudy == "selfconsistent" is False)
    - So the bundle.clouds() call is effectively dead code — the opacity used in t_start()
      never changes within a single profile() call

  New code:
    - update_clouds("fixed") returns immediately — no bundle.clouds() call
    - calculate_atm() also not re-called
    - Same effective behavior: opacity stays fixed within each profile() call

  Impact: None. Both versions have the same in-loop behavior — opacity stays constant.
  The dead bundle.clouds() call in the old code has no effect.

POTENTIAL FIX: To match the old code's behavior of ramping in the cloud at 25% on the first call,
the new code could either:
  (a) Initialize opd_cld_climate with the user cloud in ALL 4 history slots instead of just slot 0:
      This would make the 4-step average equal to the full cloud from the start.
      But this would NOT match the old behavior — it would be equivalent to 100%.
  (b) Accept the new behavior (100% from the start) as the intended design.
      The 25% ramp-in was a side-effect of the 4-step averaging starting from zeros,
      not an intentionally designed feature.
"""

# %%
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os

picaso_path = "/Users/adityasengupta/picaso"
out_dir = os.path.join(picaso_path, "figures/userdefinedclouds_comparison")
os.makedirs(out_dir, exist_ok=True)

old = pkl.load(open(os.path.join(picaso_path, 'data/bd_fixed_2602/bd_cloudmodefixed_fsed2_teff200_grav3160_semimajor0.08_prev.pkl'), 'rb'))
# %%
new = pkl.load(open(os.path.join(picaso_path, 'data/bd_fixed_2602/bd_cloudmodefixed_fsed2_teff200_grav3160_semimajor0.08.pkl'), 'rb'))

pressure = old['pressure']
nlev = len(pressure)

old_profiles = old['all_profiles'].reshape(-1, nlev)
new_profiles = new['all_profiles'].reshape(-1, nlev)
old_niters = old_profiles.shape[0]
new_niters = new_profiles.shape[0]

# ========== FIGURE 1: Overall comparison ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.semilogy(old_profiles[0], pressure, ls="--", color="gray", lw=1, label="Initial guess")
ax.semilogy(old['temperature'], pressure, lw=2, color="C0", label=f"Old final ({old_niters} iter)")
ax.semilogy(new['temperature'], pressure, lw=2, color="C1", label=f"New final ({new_niters} iter)")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (bar)")
ax.invert_yaxis()
ax.legend(fontsize=8)
ax.set_title("Final T-P profiles")

ax = axes[1]
ax.semilogx(pressure, old['temperature'] - new['temperature'])
ax.axhline(0, ls=":", color="gray")
ax.set_xlabel("Pressure (bar)")
ax.set_ylabel("Old T − New T (K)")
ax.set_title(f"Temperature difference (max = {np.max(np.abs(old['temperature'] - new['temperature'])):.0f} K)")

ax = axes[2]
for (label, profiles, color) in [("Old", old_profiles, "C0"), ("New", new_profiles, "C1")]:
    niters = profiles.shape[0]
    max_dt = [np.max(np.abs(profiles[i] - profiles[i-1])) for i in range(1, niters)]
    ax.semilogy(range(1, niters), max_dt, label=label, color=color, alpha=0.7)
ax.set_xlabel("Iteration")
ax.set_ylabel("Max |ΔT| between iterations (K)")
ax.legend()
ax.set_title("Convergence history")

plt.suptitle("g=3160, Teff=200, a=0.08 AU, fixed clouds: Old vs New", fontsize=12)
plt.tight_layout()
fname = os.path.join(out_dir, "diagnosis_overall_teff200_grav3160.png")
plt.savefig(fname, dpi=150)
plt.close()
print(f"Saved {fname}")

# ========== FIGURE 2: First few iterations showing the ramp-in effect ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show first 10 iterations for old code
ax = axes[0]
n_show = min(20, old_niters)
cmap = plt.cm.viridis(np.linspace(0, 1, n_show))
for i in range(n_show):
    ax.semilogy(old_profiles[i], pressure, color=cmap[i], alpha=0.7, lw=1)
ax.semilogy(old_profiles[0], pressure, color="red", ls="--", lw=2, label="Iteration 0 (initial)")
ax.semilogy(old['temperature'], pressure, color="black", lw=2, label="Final")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (bar)")
ax.invert_yaxis()
ax.set_title(f"Old code: first {n_show} of {old_niters} iterations")
ax.legend(fontsize=8)

# Show first 20 iterations for new code
ax = axes[1]
n_show = min(20, new_niters)
cmap = plt.cm.viridis(np.linspace(0, 1, n_show))
for i in range(n_show):
    ax.semilogy(new_profiles[i], pressure, color=cmap[i], alpha=0.7, lw=1)
ax.semilogy(new_profiles[0], pressure, color="red", ls="--", lw=2, label="Iteration 0 (initial)")
ax.semilogy(new['temperature'], pressure, color="black", lw=2, label="Final")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (bar)")
ax.invert_yaxis()
ax.set_title(f"New code: first {n_show} of {new_niters} iterations")
ax.legend(fontsize=8)

plt.suptitle("Iteration evolution: 25% ramp-in (old) vs 100% from start (new)", fontsize=12)
plt.tight_layout()
fname = os.path.join(out_dir, "diagnosis_iterations_teff200_grav3160.png")
plt.savefig(fname, dpi=150)
plt.close()
print(f"Saved {fname}")

# ========== FIGURE 3: Cloud opacity comparison ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Old code's virga output
vo = old['virga_output']
opd_old = np.array(vo['opd_per_layer'])  # shape (90, 196)
layer_p = np.sqrt(pressure[:-1] * pressure[1:])

# Sum OPD across wavelength for a simple picture
opd_old_sum = np.sum(opd_old, axis=1)

ax = axes[0]
ax.loglog(opd_old_sum, layer_p, label="Full virga OPD (100%)", color="C0", lw=2)
ax.loglog(0.25 * opd_old_sum, layer_p, label="25% virga OPD (first profile() call)", color="C0", ls="--", lw=2)
ax.set_xlabel("Summed OPD across wavelength")
ax.set_ylabel("Pressure (bar)")
ax.invert_yaxis()
ax.legend(fontsize=8)
ax.set_title("Old code: cloud opacity ramp-in")

# OPD at a single representative wavelength (index 55 ≈ 4 µm)
ax = axes[1]
wl_idx = 55
ax.loglog(opd_old[:, wl_idx], layer_p, label=f"Full virga OPD (wl idx {wl_idx})", color="C0", lw=2)
ax.loglog(0.25 * opd_old[:, wl_idx], layer_p, label=f"25% (first call)", color="C0", ls="--", lw=2)
ax.set_xlabel("OPD at ~4 µm")
ax.set_ylabel("Pressure (bar)")
ax.invert_yaxis()
ax.legend(fontsize=8)
ax.set_title("Old code: single wavelength OPD")

plt.suptitle("Cloud opacity: the 4-step averaging ramp-in effect", fontsize=12)
plt.tight_layout()
fname = os.path.join(out_dir, "diagnosis_cloud_opacity_teff200_grav3160.png")
plt.savefig(fname, dpi=150)
plt.close()
print(f"Saved {fname}")

# ========== Print summary ==========
print()
print("=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print()
print("Case: g=3160 m/s², Teff=200 K, semi_major=0.08 AU, fsed=2, fixed clouds")
print()
print(f"Old code: converged in {old_niters} iterations")
print(f"  Final T range: {np.min(old['temperature']):.1f} – {np.max(old['temperature']):.1f} K")
print(f"  Max cloud OPD: {np.max(opd_old):.1f}")
print()
print(f"New code: converged in {new_niters} iterations")
print(f"  Final T range: {np.min(new['temperature']):.1f} – {np.max(new['temperature']):.1f} K")
print()
print(f"Max |ΔT|: {np.max(np.abs(old['temperature'] - new['temperature'])):.1f} K")
print()
print("ROOT CAUSE: Two differences in the fixed-cloud code path:")
print()
print("1. CLOUD RAMP-IN (0.25x vs 1.0x on first profile() call)")
print("   Old code: update_clouds_selfconsistent() is called on the first")
print("   iteration. The 4-step averaging starts from [virga, 0, 0, 0],")
print("   producing a weighted average of 0.25 × virga_OPD. This 25% cloud")
print("   is what calculate_atm() uses for the first ~10 iterations.")
print("   After the first profile() call, the full virga OPD is used.")
print()
print("   New code: self.clouds() is called before the workflow with the full")
print("   user-provided cloud. calculate_atm() sees 100% from iteration 1.")
print()
print("   Impact: For thick clouds (max OPD ≈ 5500), starting at 25% vs 100%")
print("   pushes the T-P structure onto a fundamentally different convergence")
print("   path. The old code's gradual ramp-in appears to converge to a warmer")
print("   solution; the new code's immediate full cloud converges to a colder one.")
print()
print("2. IN-LOOP bundle.clouds() CALL (present in old, absent in new)")
print("   Old code: update_clouds('fixed_after_first') calls bundle.clouds()")
print("   every loop iteration with OPD[:,:,0] (full virga result).")
print("   BUT calculate_atm() is never re-called, so this has NO EFFECT.")
print("   New code: update_clouds('fixed') returns immediately.")
print("   Effective behavior is identical — this is NOT a contributing factor.")
print()
print("RECOMMENDED FIX:")
print("   The 25% ramp-in was a side effect of the 4-step history array")
print("   starting from zeros, not an intentional design choice. The new code's")
print("   behavior of using 100% cloud from the start is arguably more correct.")
print("   However, if matching the old behavior is desired, you could initialize")
print("   all 4 history slots with the user cloud (instead of just slot 0),")
print("   which would give 100% from the start, OR you could intentionally")
print("   scale the initial cloud to 25% for the first profile() call.")
