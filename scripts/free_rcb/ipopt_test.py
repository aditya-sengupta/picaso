import numpy as np
import cyipopt

inf = 1e20 # idk
max_temperature = 5199.0 # historical

class TemperatureOptimization():
    def __init__(self, N, log_pressure, adiabatic_slope, T0, C1, log_pressure_0, flux_threshold):
        self.N = N
        self.log_pressure = log_pressure
        self.adiabatic_slope = adiabatic_slope
        self.T0 = T0
        self.C1 = C1
        self.log_pressure_0 = log_pressure_0
        self.flux_threshold = flux_threshold
        self.constraint_bounds = None
        self.objective_bounds = [
            np.zeros((2*N-1,)),
            np.concatenate((max_temperature * np.ones(N,), np.ones(N-1,)))
        ]

    def objective(self, x):
        """Returns sum of squares of convective indicators."""
        convective_indicators = x[self.N:]
        return np.sum(convective_indicators ** 2)

    def gradient(self, x):
        """Returns gradient of objective."""
        grad = np.zeros_like(x)
        grad[self.N:] = 2 * x[self.N:]
        return grad

    def constraints(self, x):
        """Returns constraint violations."""
        temperature = x[:self.N]
        convective_indicators = x[self.N:]
        
        lapse_rate = np.diff(temperature) / np.diff(self.log_pressure)
        flux = (temperature - self.T0 - self.C1 * (self.log_pressure - self.log_pressure_0) ** 2) ** 2
        dflux_dT = np.diff(flux) / np.diff(temperature)
        
        constraints = np.concatenate([
            self.adiabatic_slope - lapse_rate, # length N-1
            np.diff(convective_indicators), # length N-2
            (1 - convective_indicators) ** 2 * dflux_dT ** 2, # length N-1
            temperature, # length N
            convective_indicators * (self.adiabatic_slope - lapse_rate), # length N-1
            np.diff(np.diff(temperature)) # length N-2
        ]) # total 6N-7
        if self.constraint_bounds is None:
            # eventually when this is stable I will move this to __init__
            self.constraint_bounds = [
                np.concatenate([
                    np.zeros((N-1,)),
                    np.zeros((N-2,)),
                    -flux_threshold * np.ones((N-1,)),
                    np.zeros((N,)),
                    np.zeros((N-1,)),
                    -0.1 * np.ones((N-2,))
                ]),
                np.concatenate([
                    inf * np.ones((N-1,)),
                    np.ones((N-2,)),
                    flux_threshold * np.ones((N-1,)),
                    max_temperature * np.ones((N,)),
                    0.1 * np.ones((N-1,)),
                    0.1 * np.ones((N-2,))
                ])
            ]
        return constraints

    def jacobian(self, x, dx=1e-8):
        constraints_up = self.constraints(x + dx)
        constraints_down = self.constraints(x - dx)
        return (constraints_up - constraints_down) / (2 * dx)

N = 91
log_pressure = np.linspace(-4, 2, N)
log_pressure_0 = np.min(log_pressure)
T0 = 100
C1 = 5
flux_threshold = 1e-30
rcb_index = 50
log_pressure_rcb = log_pressure[rcb_index]
adiabatic_slope = 2 * C1 * (log_pressure_rcb - log_pressure_0)
to = TemperatureOptimization(N, log_pressure, adiabatic_slope, T0, C1, log_pressure_0, flux_threshold)
temperature_guess = 200 + 25 * log_pressure
x0 = np.concatenate((temperature_guess, np.ones(N-1)))
cl0 = to.constraints(x0)

nlp = cyipopt.Problem(
   n=len(x0),
   m=len(cl0),
   problem_obj=to,
   lb=to.objective_bounds[0],
   ub=to.objective_bounds[1],
   cl=to.constraint_bounds[0],
   cu=to.constraint_bounds[1],
)
nlp.solve(x0)