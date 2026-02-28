I have two versions of the PICASO codebase, one at the current commit (2eacbe2) and one at a19a616. 
They should do the same thing step by step, but they are returning different results.
The test scripts for these are at scripts/correctness_test.py for the current commit and scripts/correctness_test_regression.py for the previous one.
These created the figures "comparison_teff900_grav316.png" and "regression_teff900_grav316.png", under figures/userdefinedclouds_comparison/, which show that the solution found by the model changed significantly.
Therefore there must have been an unintentional change to the program logic in the current version.
Trace through both versions and find this change. Demonstrate that reverting only whatever small piece of logic has changed would result in the same answer as before.