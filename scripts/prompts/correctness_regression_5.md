I'm not convinced that the 25% ramp-in that was the diagnosis off correctness_regression_4.md is actually happening or that it's the root cause for the convergence challenges.
I've reverted picaso to its earlier state so that we can confirm this.

Modify scripts/run_cloudy_bd_comparison.py to be compatible with the earlier API (look at scripts/bd_fixed_grid.py on the earlier commit to see what this should be), run it, and plot the OPD that PICASO actually uses per iteration during this run to show that the 25% ramp-in is having an effect.

Further, if the 25% ramp-in were the cause for the convergence challenges, removing it would cause the difficult convergence case that we're seeing now. Remove it by modifying picaso/clouds.py and rerun the test case. It doesn't need to run completely for me to see that the convergence got more difficult, so give it a 4-minute timeout. Save whatever the temperature structure is at that timeout and make updated comparison plots.