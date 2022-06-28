using ForwardDiff
using LatinHypercubeSampling
using JuliennedArrays

function Xsamples(N, Xset)
	m = size(Xset, 1)
	X = LHCoptim(N, m, 1)[1]'/N
	(X .* diff(Xset, dims=2)) .+ Xset[:,1]
end

Xset = [-1 -4; 13 4; -π π; 1.0 2.0]

h(t, x) = [0., -0.2*x[4]^2]
X = Xsamples(1000, Xset)
Δh = 1.0
Δhx = 1.0

# ╔═╡ 4d5ac286-cdc8-11ea-0791-4353d95131cb
function uncertainty_data(X::Matrix{T}, σ) where T
	_, K = size(X)
	Y = Array{T, 2}(undef, 2, K)
	for k = 1:K
		Y[:,k] .= h(0., X[:,k]) .+ σ*randn(2)
	end

	Y
end

# ╔═╡ 8aa4a55c-cd6f-11ea-3f69-75dcf0036e4e
function scaling_constants(Xset; τ=1e-8, δ=0.1)
	r = maximum(diff(Xset, dims=2))
	M = (1 + r/τ)^3
	β = 2*log(2*M/δ)
	βx = 2*log(4*M/δ)
	β, βx
end

# ╔═╡ 3deb8856-cea1-11ea-2c65-b3c12e030181
begin
	struct SEKernel{T}
		inv2L::Vector{T}
		σ2f::T
	end
	function SEKernel(params)
		# params should NOT be in log
		lscale = params[1:end-1]
		σf = params[end]
		SEKernel(lscale.^(-2), σf^2)
	end
	function (ker::SEKernel)(z)
		val = mapreduce(x->x[1]^2*x[2], +, zip(z, ker.inv2L))
		ker.σ2f*exp(-val/2)
	end

	struct GP{T}
		X::Matrix{T}
		Y::Matrix{T}
		ker::Vector{SEKernel{T}}
		A::Vector{Cholesky{T,Matrix{T}}}
		kXXy::Vector{Vector{T}}
	end
	function GP(params, X::Matrix{T}; σ = 0.01) where T
		Y = uncertainty_data(X, σ)
		n, N = size(X)
		m = length(params)
		XX = X .- reshape(X,n,1,N)
		ker = Vector{SEKernel{T}}(undef, m)
		A = Vector{Cholesky{T,Matrix{T}}}(undef, m)
		kXXy = Vector{Vector{T}}(undef, m)
		for (k,p) in enumerate(params)
			ker[k] = SEKernel(p)
			A[k] = cholesky(ker[k].(Slices(XX,1)) + (σ^2)*I)
			kXXy[k] = A[k] \ Y[k, :]
		end
		GP(X, Y, ker, A, kXXy)
	end
	function predictσ(gp::GP{T}, x) where T
		xX = x .- gp.X
		m = length(gp.ker)
		σ = Vector{T}(undef, m)
		tmp = Vector{T}(undef, size(gp.X, 2))
		for k = 1:m
			kxX = gp.ker[k].(eachcol(xX))
			ldiv!(tmp, gp.A[k], kxX)
			σ[k] = sqrt(gp.ker[k].σ2f - dot(kxX, tmp))
		end
		return σ
	end
	function predictμ(gp::GP{T}, x) where T
		m = length(gp.ker)
		μ = Vector{T}(undef, m)
		for k = 1:m
			fn(z) = z[2]*gp.ker[k](x - z[1])
			μ[k] = mapreduce(fn, +, zip(eachcol(gp.X), gp.kXXy[k]))
		end
		return μ
	end
	function predictμ!(μ, gp::GP{T}, x) where T
		for k = 1:length(μ)
			fn(z) = z[2]*gp.ker[k](x - z[1])
			μ[k] = mapreduce(fn, +, zip(eachcol(gp.X), gp.kXXy[k]))
		end
		nothing
	end
	function predict∇σ(gp::GP{T}, x) where T
		n, N = size(gp.X)
		m = size(gp.Y, 1)
		∇σ = Matrix{T}(undef, n, m)
		∇kxX = Matrix{T}(undef, N, n)
		for k = 1:m
			∇ker(x) = ForwardDiff.gradient(gp.ker[k], x)
			for i = 1:N
				∇kxX[i,:] .= ∇ker(x .- gp.X[:,i])
			end
			∇2k = Diagonal(gp.ker[k].σ2f*gp.ker[k].inv2L)
			∇σ[:,k] .= diag(sqrt(∇2k - ∇kxX'*(gp.A[k]\∇kxX)))
		end
		∇σ
	end
	function predict_error(gp::GP{T}, X, Xset) where T
		β, βx = scaling_constants(Xset)
		Δh = Δhx = 0.0

		for z in eachcol(X)
			σ = predictσ(gp, z)
            Δh = sqrt(β)*norm(σ)

			∇σ = predict∇σ(gp, z)
			Δhx = sqrt(βx)*opnorm(∇σ)
		end

		Δh, Δhx
	end
	improvement(prev, next) = (prev - next)/prev*100
end

xparams = [[1e8, 1e8, 1e8, 1e8, 1.3],[1e8, 1e8, 1e8, 3.0, 1.3]]
xdata = Xsamples(50, Xset)
gp = GP(xparams, xdata)
Δh, Δhx = predict_error(gp, Xsamples(1000, Xset), Xset)
vN(x) = predictμ(gp, x)
#  μ = zeros(2)
#  z = rand(3)
