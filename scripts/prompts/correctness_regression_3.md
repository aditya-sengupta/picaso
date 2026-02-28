I'm running fixed cloud models and noticing some significant differences in the convergence speed and results between different versions of the feature.
I've made some intended changes and want to make sure those fully explain these differences.
1. I changed the initial temperature and the corresponding cloud structure
2. I changed the behavior at the first iteration so that it always sees the fixed cloud, instead of running it within the climate model and reusing that solution
3. It's possible I added in something else that makes this incorrect.

I'd like to look at the case with g = 3160, teff = 200, semi_major = 0.08. I have a pickled version of the previous output under data/bd_fixed_2602/bd_cloudmodefixed_fsed2_teff200_grav3160_semimajor0.08_prev.pkl, and the new version is at data/bd_fixed_2602/bd_cloudmodefixed_fsed2_teff200_grav3160_semimajor0.08.pkl

I believe the old version was run using commit a19a616, but I'm not certain. I'm also not certain what initial temperature was assumed. Here's the steps to track down the root cause:

1. Unpickle both outputs and compare their temperature structures directly. Compare both the final structures and the initial guesses (the first 91 elements of 'all_profiles'). 
2. Make a plot comparing these with the initial guesses in dashed lines, and save it under "figures/userdefinedclouds_comparison/".
3. Copy "run_cloudy_bd.py" to a new diagnostic script. Modify it to match the parameters I've listed instead of looking at sys.argv, and also modify it so that the filename stem has a tag like "comparison" to avoid overwrites.
4. I will then run this model myself. Provide an additional script to make a plot comparing this new run to the other two. The script should save this figure under "figures/userdefinedclouds_comparison/" as well.

Do not run any Git commands that modify the picaso/ directory, e.g. checkouts, commits, merges.