function coeffs(γ, x0, xf, A, T)
    Ti = @view T[:,2:end-1]
    z = A*[2*Ti*γ'; x0'; xf']
    (@view z[1:end-2,:])'
end

function energy(c, T, Ts, weights, W)
    E = 0.0
    x = c*T
    γs = c*Ts
    @inbounds for k = 1:length(weights)
        xv = view(x,:,k)
        γsv = view(γs,:,k)
        E += γsv'*(W(xv)\γsv)*weights[k]
    end
    return E
end

function energy(γ, x0, xf, A, T, Ts, weights, W)
    c = coeffs(γ, x0, xf, A, T)
    energy(c, T, Ts, weights, W)
end

function _u_ccm(integ)
    @unpack f, B = integ.p[1]
    @unpack xs, us, λ, W, A, T, Ts, w, γ0 = integ.p[2]
    t = integ.t
    x = integ.u.x

    xt = xs(t)
    ut = us(t)

    obj(γ) = energy(γ, xt, x, A, T, Ts, w, W)
    ret = optimize(obj, γ0, BFGS(); autodiff=:forward)
    γ0 .= Optim.minimizer(ret)
    E = minimum(ret)
    c = coeffs(γ0, xt, x, A, T)

    γs1 = c*(@view Ts[:,end])
    γs0 = c*(@view Ts[:,1])
    lhs = 2*(W(x)\B(x))'*γs1
    nlhs = sum(abs2, lhs)
    rhs = -2*λ*E - 2*γs1'*(W(x)\(f(x) + B(x)*ut)) + 2*γs0'*(W(xt)\(f(xt) + B(xt)*ut))
    if (rhs > 0) || (nlhs == 0)
        integ.p[3].uc = ut
    else
        integ.p[3].uc =  ut + rhs*lhs/nlhs
    end
end

function _u_ccm(x, sys, ccm::FlatCCMParams, t)
    @unpack f, B = sys
    @unpack xs, us, λ, M = ccm
    xt = xs(t)
    ut = us(t)
    γs = x - xt
    E = γs'*M*γs
    lhs = 2*B(x)'*M*γs
    nlhs = sum(abs2, lhs)
    rhs = -2*λ*E - 2*γs'*M*(f(x) - f(xt) + (B(x) - B(xt))*ut)
    if (rhs > 0) || (nlhs == 0)
        return ut
    else
        return ut + rhs*lhs/nlhs
    end
end
