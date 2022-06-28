using Flux
using BSON: @load
using LinearAlgebra
using DifferentialEquations
using Plots
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using SafeSimultaneousLearningControl
global_logger(TerminalLogger())

function torchparams(pkl)
    torch_1_w = pkl["0.weight"]
    torch_1_b = pkl["0.bias"]
    torch_2_w = pkl["2.weight"]
    torch_params = params(torch_1_w, torch_1_b, torch_2_w)
end

@load "data/car_W.bson" nn_W
@load "data/car_Wbot.bson" nn_Wbot

function W_func(x)
    Weval = reshape(nn_W(x[3:4]), 4, 4)
    Weval[1:2, 1:2] .= reshape(nn_Wbot([1]), 2, 2)
    Weval[1:2, 3:end] .= 0
    Weval*Weval' + 0.5I
end

# Define system matrices or functions
f(x) = [x[4]*cos(x[3]), x[4]*sin(x[3]), 0., 0.]
B = [0. 0.; 0. 0.; 1. 0.; 0. 1.]

# Simulation time span
tspan = (0., 8.)

# Intial condition of the system
x0 = [0.5, 0.0, 1.0, 1.0]

# Define desired regulation point or trajectories (as functions)
xs(t) = [4-3*cos(0.4*t), 3*sin(0.4*t), pi/2 - 0.4*t, 1.2]
us(t) = [-0.4, 0.0]
λ = 0.4

# Uncertainty unknown to the controller
h(t, x) = [0., -0.1*x[4]^2]
Δh = 0.4
ω = 50
Γ = 4e7

sys_p = sys_params(f, B)
ccm_p = ccm_params(xs, us, λ, W_func)
l1_p = l1_params(ω, Γ, Δh)

nom_sys = nominal_system(sys_p, ccm_p; Ts = 0.002)
nom_sol = solve(nom_sys, x0, tspan, Rodas4(), progress = true, progress_steps = 1)

#  l1_sys = l1_system(sys_p, ccm_p, l1_p, h)
#  l1_sol = solve(l1_sys, x0, tspan, Rosenbrock23(), progress = true, progress_steps = 1, saveat = 0.01)

plot(nom_sol, vars=[(1,2)])
traj = hcat(xs.(tspan[1]:0.1:tspan[2])...)'
plot!(traj[:,1], traj[:,2], line = (1, :dash, :red), legend=:none)
