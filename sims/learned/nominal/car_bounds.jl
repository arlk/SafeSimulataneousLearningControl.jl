# Control
using Flux
using BSON: @load
using ForwardDiff
using LatinHypercubeSampling

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
λ = 0.5
αupp = 2.0
αlow = 1.0
Δus = 1.12
ϵ = 0.2
ρr = 0.5*sqrt(αupp/αlow) + ϵ # sqrt(2)*0.5 + 0.4
ρa = 0.01
ρ = ρr + ρa # sqrt(2)*0.5 + 0.4 + 0.1
Δh = 1.0
Δhx = 1.0

∇f(x) = ForwardDiff.jacobian(f, x)

function δu(W, λ, B, x)
    ∇W(x) = ForwardDiff.jacobian(W, x)

    ∇fx = ∇f(x)
    ∇Wx = ∇W(x)
    fx = f(x)
    Wx = W(x)

    n = length(x)

    Fx = -reshape(∇Wx[:,3].*fx[3] .+ ∇Wx[:,4].*fx[4], n, n)
    Fx .+= ∇fx*Wx .+ Wx*∇fx' .+ 2*λ*Wx

    Li = inv(sqrt(Wx))

    num = eigmax(Li'*Fx*Li)
    svals = eigvals(B'*Li*Li'*B)
    den = sqrt(minimum(svals[svals .> 0]))

    num/den/2.0
end

function normMx(W, x)
    ∇M(x) = ForwardDiff.jacobian(x->inv(W(x)), x)
    ∇Mx = ∇M(x)
    n = length(x)
    ret = opnorm(reshape(∇Mx[:,3], n, n))
    ret + opnorm(reshape(∇Mx[:,4], n, n))
end

function Xsamples(N, Xset)
	m = size(Xset, 1)
	X = LHCoptim(N, m, 1)[1]'/N
	(X .* diff(Xset, dims=2)) .+ Xset[:,1]
end

Xset = [0 0; 0 0; -π π; 1.0 2.0]

Ω = Xsamples(1000, Xset)

ΔB = opnorm(B)

Δf = maximum(x->norm(f(x)), eachcol(Ω))

Δδu = maximum(x->δu(W_func, λ, B, x), eachcol(Ω))

Δxr = Δf + ΔB*(Δh + Δus + ρ*Δδu)

ΔMx = maximum(x->normMx(W_func, x), eachcol(Ω))

ΔΨx = ΔB*ΔMx/αlow

Δfx = maximum(x->opnorm(∇f(x)), eachcol(Ω))

Δγ = sqrt(αupp/αlow)*(Δfx + Δhx*ΔB) + Δδu*ΔB

Δx = Δf + ΔB*(2*Δh + Δus + ρ*Δδu)

ΔΨ = (Δγ + ΔMx*Δx/sqrt(αupp*αlow))*ΔB*αupp

ΔBp = opnorm(pinv(B))

Δxt = sqrt(2*Δh*(Δhx*Δx) + 4*Δh^2)

Δη(ω) = (ω + 1)*ΔBp*Δxt

Δθ(ω) = ΔB*Δη(ω)*αupp/λ

ζ1(ω) = 2*ρ*ΔB*(αupp/αlow)*(Δh/abs(2*λ - ω) + (Δhx*Δxr)/(2*λ*ω))

ζ2(ω) = ΔΨx*αupp*(Δh/abs(2*λ - ω) + (Δhx*Δxr)/(2*λ*ω))

ζ3(ω) = Δhx*αupp*(4*λ*ΔB + ΔΨ)/(2*λ*ω)
