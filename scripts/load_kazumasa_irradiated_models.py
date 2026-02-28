import os
import numpy as np

def multidimensional_interpolation_weights(
    params, 
    params_lower, 
    params_upper
):
    """
    Returns the linear combination of grid points that provide an interpolation of some function of the parameters.
    
    Suppose we want to estimate f(*params) and we know the value of f on each combination of params_lower and params_upper in each argument. This function returns the weights by which we should multiply each of those known values such that we get f(*params).
    
    Returns an array of 2^n weights, where weight i corresponds to the grid point whose binary representation is bin(i) interpreted such that 0 is the lower value and 1 is the upper value. For 3 arguments, this order would be:
    
    l, l, l
    l, l, u
    l, u, l
    l, u, u
    u, l, l
    u, l, u
    u, u, l
    u, u, u
    """
    params_upper = np.copy(params_upper)
    assert (n := len(params)) == len(params_lower) == len(params_upper)
    for i in range(n):
        if params_lower[i] == params_upper[i]:
            assert params[i] == params_lower[i]
            params_upper[i] += 1
            
    interpolation_distances = np.array([(p - pl) / (pu - pl) for (p, pl, pu) in zip(params, params_lower, params_upper)])
    assert np.all(0 <= interpolation_distances) and np.all(interpolation_distances <= 1)
    weights = np.ones(2**n)
    for i in range(2**n):
        for (j, x) in enumerate(bin(i)[2:].zfill(n)):
            if int(x) == 0:
                weights[i] *= 1 - interpolation_distances[j]
            else:
                weights[i] *= interpolation_distances[j]
                
    return weights

grid_path = "/Users/adityasengupta/picaso/models_kazumasa"

grid_values = [
    np.array([1, 3, 10, 31, 100]), # metallicity
    np.array([6, 7, 8, 9]), # x
    np.arange(1, 12), # y
    np.arange(1, 6) # z
]
grid_min = [np.min(x) for x in grid_values]
grid_max = [np.max(x) for x in grid_values]

def kazumasa_hj_grid_interpolation(metallicity, semimajor_axis_au, tint_K, grav_ms2, pressure_grid=np.logspace(-6, 2, 91)):
    """
    $[M/H]   = 0, 0.5, 1.0, 1.5, 2.0 $

    Semi-major axis = $1170\times 10^{-x/2}$ AU (for Sun, x=6,7,8,9)

    $T_{\rm int} = 24\times 10^{(y-1)/5} (y=1,2,...,11)$

    $g       = 0.01\times10^{z} {\rm m/s^2}$ (z=1,2,3,4,5)

    ## The file name follows the format of "PT_Z($10^{[M/H]}$)_Teq($x$)_Tin($y$)_g($z00$).dat"
    Example) "PT_Z10_Teq10_Tin011_g100.dat" $\rightarrow$ x=10, y=11, z=1
    """
    # first, convert the parameters we've passed in to grid x/y/z
    params = [
        metallicity,
        -2*np.log10(semimajor_axis_au / 1170), # x
        1 + 5*np.log10(tint_K / 24), # y
        np.log10(grav_ms2 / 0.01) # z
    ]
    # next, if any params are outside the range, truncate them
    params = np.clip(params, grid_min, grid_max)
    # now, search for the upper and lower values on each parameter that are closest
    all_lower_params = [[x for x in grid_values[i] if x <= params[i]] for i in range(4)]
    all_upper_params = [[x for x in grid_values[i] if x > params[i]] for i in range(4)]
    params_lower = [
        np.max(all_lower_params[i])
        for i in range(4)
    ]
    params_upper = [
        np.min(all_upper_params[i]) if len(all_upper_params[i]) > 0 else grid_values[i][-1]
        for i in range(4)
    ]
    
    # get the weights for these parameter ranges
    weights = multidimensional_interpolation_weights(params, params_lower, params_upper)
    res = np.zeros(91)
    for (i, w) in enumerate(weights):
        if w > 0:
            x = bin(i)[2:].zfill(4)
            params_to_load = [str(params_lower[j]) if int(x[j]) == 0 else str(params_upper[j]) for j in range(4)]
            fname = f"PT_Z{params_to_load[0]}_Teq{params_to_load[1].zfill(2)}_Tin{params_to_load[2].zfill(3)}_g{params_to_load[3]}00.dat"
            data = np.loadtxt(
                    os.path.join(grid_path, fname), 
                    skiprows=4
                )
            # not all the pressure grids are the same + some of them shoot past the 5199 K PICASO limit, so we're going to re-interpolate them over `pressure_grid`
            interpolated_temperature = np.interp(pressure_grid, data[:,1], data[:,2])
            res += w * interpolated_temperature
        
    return res
            