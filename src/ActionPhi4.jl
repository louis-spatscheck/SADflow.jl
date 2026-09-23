# src/ActionPhi4.jl

"""
    Phi4Params{T}

Bare couplings of the 2D lattice φ⁴ theory.
- `κ` : hopping parameter
- `λ` : quartic coupling
"""
struct Phi4Params{T}
    κ::T
    λ::T
end

"""
    action(x, params)

Lattice φ⁴ action density summed over spatial dims (1,2,3):

    S = Σ [ -2κ φ·(neighbour sum) + φ² + λ(φ²-1)² ]
"""
function action(x, params::Phi4Params)
    κ = params.κ
    λ = params.λ
    return sum(
        -2 * κ.* x .* neighbour_sum(x) 
        .+ x .^ 2 
        .+
        λ .* (x .^ 2 .- 1.0) .^ 2
        ,dims=(1, 2, 3)
        )
end

"""
    hessian(phi, f, params)

Local Hamiltonian-shift term H(φ, f) used inside `KLloss`.
Summed over the spatial+channel dims (1,2,3). 
"""
function hessian(phi, f, params::Phi4Params)
    κ = params.κ
    λ = params.λ
    interaction = -2 * κ .* (f .* staple(f))
    prefac      = (2 - 4 * λ) .+ 12 * λ .* (phi .^ 2)
    potential   = prefac .* (f .^ 2)

    H_local = interaction .+ potential
    return sum(H_local, dims=(1, 2, 3))
end

"""
    source_derivative(phi)
Source term Σ_x φ(t=1, x).
"""
function source_derivative(phi)
    return sum(phi[1, :, :, :], dims=(1, 2))
end