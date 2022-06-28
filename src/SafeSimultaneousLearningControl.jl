module SafeSimultaneousLearningControl

using ComponentArrays
using LinearAlgebra
using Optim
using OrdinaryDiffEq
using DiffEqCallbacks
using UnPack

import DiffEqBase: solve

export sys_params, ccm_params, l1_params
export nominal_system, reference_system, l1_system

include("chebyshev.jl")
include("types.jl")
include("sys.jl")
include("ccm.jl")
include("l1.jl")

end # module
