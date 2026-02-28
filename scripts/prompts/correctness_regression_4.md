The test in correctness_regression_3.md showed that the two versions of the algorithm return different results even if they're at the same starting point. 
I've made minor changes to how the algorithm should function, but I don't expect there to have been big enough changes to cause this.
Trace through the previous versions of the code and the current version, and diagnose the change.

You may copy code into scripts/ but you may not change anything in the picaso/ directory or change the Git state. If you need to rerun old versions of the model, run it as a copied version from scripts/.

At the end, create a script explaining the difference so that I can make any necessary fixes to picaso/ myself. This script should create explanatory figures that are saved in "figures/userdefinedclouds_comparison".