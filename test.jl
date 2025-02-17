using Revise
using MPSKit, MPSKitModels, TensorKit

D = 4 # bonddimension
init_state = InfiniteMPS(ℂ^2, ℂ^D)

g = 0.5

H = transverse_field_ising(; g=g)
x = RecordEnergyConvergence()
groundstate, environment, δ = find_groundstate(init_state, H, VUMPS(; verbosity=3, tol = 1e-10, finalize = x))

(;energies) = x.data
scatter(g_values, M; xlabel="g", ylabel="M", label="D=$D", title="Magnetization")