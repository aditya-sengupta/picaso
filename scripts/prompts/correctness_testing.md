I have two versions of a feature that adds "fixed" clouds to a climate calculation, and I want to see if they're the same.
Part of my implementation is when I calculate the cloud profile that I'm fixing, and I recently changed this so that users could provide arbitrary input.
However, since this changed the cloud profile I was fixing, I'm no longer sure of the correctness of the feature.
 
I have a suggested way of testing this:
1. Load the output from the old version; these are saved for a range of parameters under "data/bd_fixed_2602/*.pkl".
2. Look up the cloud profile that was used within these old runs.
3. Carry out a new run using this cloud profile.
4. Plot the temperature structures from both versions (panel [0, 1] of diagnostic_plot)

Do this for three randomly chosen parameters out of the old outputs. Save the resulting comparison plots to "figures/userdefinedclouds_comparison/".

