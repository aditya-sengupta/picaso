# import nimbus
import numpy as np
from nimbus import Nimbus

# define temperature [K] and pressure [bar] structure
temperature = np.asarray([554, 572, 607, 653, 775, 951, 1073, 1111, 1540, 2654, 3000])
pressure = np.asarray([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

# set up Nimbus object
obj = Nimbus()
obj.set_up_atmosphere(
    temperature = temperature,
    pressure = pressure,
    kzz = np.ones_like(pressure) * 1e9,
    mmw = 2.34,
    gravity = 10**2.49,
    species = 'SiO',
    deep_mmr = 10**-3,
)
obj.set_up_solver()

# compute the cloud structure
ds = obj.compute()