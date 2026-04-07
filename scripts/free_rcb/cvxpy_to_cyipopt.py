import numpy as np
import cyipopt
import cvxpy as cp

# ══════════════════════════════════════════════════════════════════════
# Problem data (shared)
# ══════════════════════════════════════════════════════════════════════
N = 91
log_pressure = np.linspace(-4, 2, N)
log_pressure_0 = np.min(log_pressure)
dlogP = np.diff(log_pressure)  # shape (N-1,)

T0 = 100
C1 = 5
flux_threshold = 1e-30
rcb_index = 50
log_pressure_rcb = log_pressure[rcb_index]
adiabatic_slope = 2 * C1 * (log_pressure_rcb - log_pressure_0)

# Equilibrium profile: a[i] = T0 + C1*(logP[i] - logP_0)^2
a = T0 + C1 * (log_pressure - log_pressure_0) ** 2

# ══════════════════════════════════════════════════════════════════════
# 1) CVXPY version
# ══════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CVXPY solve")
print("=" * 60)

target_temperature = np.zeros((N,))
target_temperature[:rcb_index] = T0 + C1 * (log_pressure[:rcb_index] - log_pressure_0) ** 2
target_temperature[rcb_index:] = T0 + C1 * (log_pressure[rcb_index] - log_pressure_0) ** 2 + adiabatic_slope * (log_pressure[rcb_index:] - log_pressure[rcb_index])
temperature = cp.Variable(shape=(N,))
convective_indicators = cp.Variable(shape=(N-1,), bounds=[0,1])

lapse_rate = cp.diff(temperature) / np.diff(log_pressure)
lapse_rate_constraint = lapse_rate <= adiabatic_slope
convective_below_constraint = cp.diff(convective_indicators) >= 0
flux = (temperature - T0 - C1 * (log_pressure - log_pressure_0) ** 2) ** 2
dflux_dT = cp.diff(flux) / cp.diff(temperature)
flux_balance_constraint = cp.square(cp.multiply((1 - convective_indicators), dflux_dT)) <= flux_threshold
temperature_positive_constraint = temperature >= 0
convective_lapse_rate_constraint = cp.multiply(convective_indicators, (adiabatic_slope - lapse_rate)) <= 0.1
temperature_smoothness_constraint = cp.diff(cp.diff(temperature)) <= 0.1
constraints = [lapse_rate_constraint, convective_below_constraint, flux_balance_constraint, temperature_positive_constraint, temperature_smoothness_constraint, convective_lapse_rate_constraint]

problem = cp.Problem(cp.Minimize(cp.sum_squares(convective_indicators)), constraints)

problem.solve(solver=cp.IPOPT, nlp=True,
              hessian_approximation='limited-memory')

cvxpy_T = temperature.value
cvxpy_c = convective_indicators.value
print(f"CVXPY objective : {problem.value}")
print(f"CVXPY status    : {problem.status}")

# ── Extract the exact data CVXPY passes to cyipopt ────────────────────
# Re-run the reduction chain (without solving) to get the Bounds object.
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
from cvxpy.reductions.solvers.nlp_solvers.ipopt_nlpif import IPOPT as IPOPT_nlp
from cvxpy.reductions.solvers.solving_chain import SolvingChain
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver, Bounds

# Reset variable values to what set_NLP_initial_point would produce
# (before the first solve, not the solved values)
for var in problem.variables():
    var.save_value(None)
problem.set_NLP_initial_point()
print(f"temperature init : {temperature.value[:5]} ...")
print(f"conv_ind init    : {convective_indicators.value[:5]} ...")

nlp_chain = SolvingChain(reductions=[
    CvxAttr2Constr(reduce_bounds=False), Dnlp2Smooth(), IPOPT_nlp()
])
canon_data, inverse_data = nlp_chain.apply(problem=problem)

# canon_data is the dict from NLPsolver.apply()
cvxpy_x0 = canon_data['x0']
cvxpy_lb = canon_data['lb']
cvxpy_ub = canon_data['ub']
cvxpy_cl = canon_data['cl']
cvxpy_cu = canon_data['cu']
print(f"CVXPY x0 shape   : {cvxpy_x0.shape}")
print(f"CVXPY lb range   : [{cvxpy_lb.min():.2e}, {cvxpy_lb.max():.2e}]")
print(f"CVXPY ub range   : [{cvxpy_ub.min():.2e}, {cvxpy_ub.max():.2e}]")

# ══════════════════════════════════════════════════════════════════════
# 2) Direct cyipopt version — using CVXPY's Oracles (C diff engine)
# ══════════════════════════════════════════════════════════════════════
#
# Instead of re-implementing the Dnlp2Smooth decomposition by hand,
# we use CVXPY's own Oracles object (wrapping the C-based diff engine)
# to provide the exact same objective, gradient, constraints, and Jacobian
# that CVXPY passes to cyipopt internally.  The only difference is that
# WE construct the cyipopt.Problem and call nlp.solve() ourselves.
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Direct cyipopt solve (using CVXPY Oracles)")
print("=" * 60)

from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import Oracles

bounds_obj = canon_data['_bounds']
n_vars = len(cvxpy_x0)
n_con = len(cvxpy_cl)

oracles = Oracles(bounds_obj.new_problem, cvxpy_x0, n_con,
                  verbose=True, use_hessian=False)

nlp = cyipopt.Problem(
    n=n_vars, m=n_con,
    problem_obj=oracles,
    lb=cvxpy_lb, ub=cvxpy_ub, cl=cvxpy_cl, cu=cvxpy_cu,
)
nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-7)
nlp.add_option('bound_relax_factor', 0.0)
nlp.add_option('hessian_approximation', 'limited-memory')
nlp.add_option('derivative_test', 'none')
nlp.add_option('least_square_init_duals', 'yes')
nlp.add_option('print_level', 2)

x_opt, info = nlp.solve(cvxpy_x0)

# Extract T and c from x_opt using the reduced problem's variable layout
reduced_vars = bounds_obj.new_problem.variables()
var_offsets = {}
off = 0
for v in reduced_vars:
    var_offsets[v.name()] = (off, v.shape, v.size)
    off += v.size

# Find temperature and convective_indicators in the reduced problem
T_opt = None
c_opt = None
for v in reduced_vars:
    voff, vshape, vsize = var_offsets[v.name()]
    # Match by shape — temperature is (91,), convective_indicators is (90,)
    if vsize == N and v.name() == temperature.name():
        T_opt = x_opt[voff:voff+vsize]
    elif vsize == N - 1 and v.name() == convective_indicators.name():
        c_opt = x_opt[voff:voff+vsize]

# Fallback: find by shape if names don't match after reductions
if T_opt is None or c_opt is None:
    for v in reduced_vars:
        voff, vshape, vsize = var_offsets[v.name()]
        if T_opt is None and vsize == N and v.is_nonneg():
            T_opt = x_opt[voff:voff+vsize]
        elif c_opt is None and vsize == N - 1:
            bnds = v.get_bounds()
            if bnds is not None:
                lb, ub = bnds
                if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
                    c_opt = x_opt[voff:voff+vsize]

print(f"\ncyipopt objective: {info['obj_val']}")
print(f"cyipopt status   : {info['status']} {info['status_msg']}")
print(f"cyipopt iters    : {oracles.iterations}")

# ══════════════════════════════════════════════════════════════════════
# 3) Compare results
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)

# Restore solved values on the original variables for comparison
# The CVXPY solve already set temperature.value and convective_indicators.value
# We need to extract the same from x_opt using the reduced problem's variable map
# Add 'iterations' key that CVXPY's invert expects
info['iterations'] = oracles.iterations
reduced_vars = bounds_obj.new_problem.variables()
off = 0
for v in reduced_vars:
    vsize = v.size
    v.save_value(np.reshape(x_opt[off:off+vsize], v.shape, order='F'))
    off += vsize

# Now use the inverse_data to map back to original variables
solution_from_direct = nlp_chain.invert(info, inverse_data)
if solution_from_direct.status in ('optimal', 'optimal_inaccurate'):
    T_direct = solution_from_direct.primal_vars.get(temperature.id)
    c_direct = solution_from_direct.primal_vars.get(convective_indicators.id)
    if T_direct is not None and c_direct is not None:
        print(f"Objective  — CVXPY: {problem.value:.8f}  cyipopt: {info['obj_val']:.8f}")
        print(f"T max-diff : {np.max(np.abs(cvxpy_T - T_direct)):.2e}")
        print(f"c max-diff : {np.max(np.abs(cvxpy_c - c_direct)):.2e}")
    else:
        print(f"Objective  — CVXPY: {problem.value:.8f}  cyipopt: {info['obj_val']:.8f}")
        print("Could not extract T/c from direct solve for detailed comparison.")
else:
    print(f"Direct solve status: {solution_from_direct.status}")
    print(f"Objective  — CVXPY: {problem.value:.8f}  cyipopt: {info['obj_val']:.8f}")
