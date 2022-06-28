# Control
using Flux
using BSON: @load
using ForwardDiff
using LatinHypercubeSampling

@load "quad_W.bson" nn_W
@load "quad_Wbot.bson" nn_Wbot

function W_func(x)
    Weval = reshape(nn_W(x[4:8]), 8, 8)
    Weval[1:5, 1:5] .= reshape(nn_Wbot(x[4:5]), 5, 5)
    Weval[1:5, 6:end] .= 0
    Weval*Weval' + 0.1I
end

# Define system matrices or functions
function f(x)
    [x[4], x[5], x[6], 9.81*tan(x[7]), 9.81*tan(x[8]), 0., 0., 0.]
end
B = [0. 0. 0.; 0. 0. 0.; 0. 0. 0.;
     0. 0. 0.; 0. 0. 0.;
     1. 0. 0.; 0. 1. 0.; 0. 0. 1.]
λ = 0.1
αupp = 5.0
αlow = 1.0
Δus = 2.0
ϵ = 0.2
ρr = 0.1*sqrt(αupp/αlow) + ϵ # sqrt(2)*0.5 + 0.4
ρa = 0.01
ρ = 1
Δh = 0.15
Δhx = 0.2

∇f(x) = ForwardDiff.jacobian(f, x)

function δu(W, λ, B, x)
    ∇W(x) = ForwardDiff.jacobian(W, x)

    ∇fx = ∇f(x)
    ∇Wx = ∇W(x)
    fx = f(x)
    Wx = W(x)

    n = length(x)

    Fx = -reshape(∇Wx[:,4].*fx[4] .+ ∇Wx[:,5].*fx[5], n, n)
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
    ret = 0
    for i = 4:8
        ret += opnorm(reshape(∇Mx[:,i], n, n))
    end
    return ret
end

function Xsamples(N, Xset)
	m = size(Xset, 1)
	X = LHCoptim(N, m, 1)[1]'/N
	(X .* diff(Xset, dims=2)) .+ Xset[:,1]
end

function tube(x::Array{T, 2}, ρ) where {T}
	n = size(x, 2)
	K = size(x, 1)
	Ω = Array{T, 2}(undef, n, K)
	for k = 1:K
        Ω[:, k] .= x[k,:] + ρ*normalize(randn(n))*(rand()^(1/n))
	end
	Ω
end

Xset = [0 0; 0 0; 0 0; -2 2; -2 2; -2 2; -π/3 π/3; -π/3 π/3]

#  Ω = Xsamples(1000, Xset)
Ω = tube(Array(traj), 1.0)

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
