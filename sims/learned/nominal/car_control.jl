# Control
using Flux
using BSON: @load
using Interpolations
using DifferentialEquations
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

@load "car_W.bson" nn_W
@load "car_Wbot.bson" nn_Wbot

function W_func(x)
    Weval = reshape(nn_W(x[3:4]), 4, 4)
    Weval[1:2, 1:2] .= reshape(nn_Wbot([1]), 2, 2)
    Weval[1:2, 3:end] .= 0
    Weval*Weval' + 0.5I
end

# Define system matrices or functions
f(x) = [x[4]*cos(x[3]), x[4]*sin(x[3]), 0., 0.]
B = [0. 0.; 0. 0.; 1. 0.; 0. 1.]

# Define desired regulation point or trajectories (as functions)
function desired_traj(solver)
    n, m = size(solver.solver_al.solver_uncon.model)
    tf = solver.solver_al.solver_uncon.tf
    N = solver.solver_al.solver_uncon.N
    t0 = range(0.0, tf, length=N)
    xs = reduce(hcat, states(solver))'
    xitp = Interpolations.scale(interpolate(xs,
               (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), t0, 1:n)
    xf(t) = xitp.(Ref(t),1:n)
    if m == 1
        us = reduce(vcat, controls(solver))
        us = vcat(us, 0.0)
        uf = Interpolations.scale(interpolate(us,
                   BSpline(Cubic(Natural(OnGrid())))), t0)
    else
        us = reduce(hcat, controls(solver))'
        us = vcat(us, zeros(1,m))
        uitp = Interpolations.scale(interpolate(us,
                   (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), t0, 1:m)
        uf(t) = uitp.(Ref(t),1:m)
    end
    return xf, uf
end
xs, us = desired_traj(solver)
λ = 0.4

# Uncertainty unknown to the controller
h(t, x) = [0., -0.2*x[4]^2]
Δh = 1.0
Γ = 1e6
ω = 90

sys_p = sys_params(f, B)
ccm_p = ccm_params(xs, us, λ, W_func)
l1_p = l1_params(ω, Γ, Δh)

l1_sys = l1_system(sys_p, ccm_p, l1_p, h)
