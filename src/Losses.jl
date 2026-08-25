# src/losses.jl
#


# --- Trace estimators ---------------------------------------------------

"""
    vjv(rng, func, z)

Single Hutchinson vector-Jacobian-vector sample: `η' J η` for a Rademacher
vector `η`, computed via a ForwardDiff directional derivative.
"""
function vjv(rng, func, z)
    T = eltype(z)
    η  = T.(rand(rng, [-1, 1], size(z)...))
    Jη = ForwardDiff.derivative(t -> func(z .+ t .* η), zero(T))
    return eachslice(η .* Jη, dims=ndims(η)) .|> sum
end

trJ(rng, func, z; ns=1) = [vjv(rng, func, z) for _ in 1:ns] |> stack
trJ(func, z; ns=1) = trJ(Random.default_rng(), func, z; ns=ns)

"""
    vjjv(rng, func, z)

Hutchinson estimator for `η' (J'J) η`, i.e. the squared-Jacobian trace
term, via two nested ForwardDiff directional derivatives.
"""
function vjjv(rng, func, z)
    T = eltype(z)
    η   = T.(rand(rng, [-1, 1], size(z)...))
    Jη  = ForwardDiff.derivative(t -> func(z .+ t .* η), zero(T))
    JJη = ForwardDiff.derivative(t -> func(z .+ t .* Jη), zero(T))
    return eachslice(η .* JJη, dims=ndims(η)) .|> sum
end

function trJJ(rng, func, z; ns=1)
    raw = [vjjv(rng, func, z) for _ in 1:ns] |> stack
    return mean(raw, dims=2)
end
trJJ(func, z; ns=1) = trJJ(Random.default_rng(), func, z; ns=ns)

"""
    ∇f_sq(x, F; K=5)

Zygote-pullback-based Hutchinson estimator of `mean(‖J v‖²)` for random
`v`, used as an alternative to the ForwardDiff-based `trJJ` above. Returns
`(f(x), estimate)`.
"""
function ∇f_sq(x::AbstractArray{T,Nd}, F; K=5) where {T,Nd}
    (X, Y, c, N) = size(x)
    tr2 = zero(T)
    f, back = Zygote.pullback(z -> F(z), x)
    for k in 1:K
        v  = randn(T, size(x))
        Jv = back(v)[1]
        tr2 += sum(Jv .* Jv)
    end
    return f, tr2 / K / (N * c)
end

"""
    ∇f_sq_exact(x, F)

Exact (non-stochastic) version of the above via a full `ForwardDiff.jacobian`
per sample in the batch.
"""
function ∇f_sq_exact(x::AbstractArray{T,4}, F) where T
    (X, Y, c, N) = size(x)
    f = F(x)
    tr2 = zero(T)
    D = X * Y * c
    f_single = z -> vec(F(reshape(z, X, Y, c, 1))[:, :, :, 1])
    for i in 1:N
        J = ForwardDiff.jacobian(f_single, vec(x[:, :, :, i]))
        tr2 += sum(J .^ 2)
    end
    return f, tr2 / (N * c)
end


# --- KL-divergence losses ------------------------------------------------

"""
    KLloss(z, F, σ², params; K=5, analytics=false)

Per-sample-averaged KL loss (scalar). `analytics=true` also returns the
three individual terms for logging/debugging.
"""
function KLloss(z::AbstractArray{T,Nd}, F, σ², params::Phi4Params; K=5, analytics=false) where {T,Nd}
    func = ModelWrapper(F)
    f_z = func(z)

    f1 = Zygote.ignore() do
        mean(trJJ(func, z; ns=K))
    end

    f2 = mean(hessian(z, f_z, params))
    f3 = -2.0 * mean(source_derivative(f_z))

    if analytics
        return 0.5f0 * (σ² + f1 + f2 + f3), f1, f2, f3
    end
    return 0.5f0 * (σ² + f1 + f2 + f3)
end

"""
    KLloss_batch(z, F, σ², params; K=5)

Batched (non-reduced-to-scalar-until-the-end) KL loss — this is the one
actually used as `loss_function` in the `training_5.jl` training loop.
"""
function KLloss_batch(z, F, σ², params::Phi4Params; K=5)
    func = ModelWrapper(F)
    f_z = func(z)

    f1 = mean(trJJ(func, z; ns=K), dims=2)
    

    f2 = hessian(z, f_z, params)
    f3 = -2.0 * source_derivative(f_z)

    return 0.5 * mean(f1 .+ f2 .+ f3 .+ σ²)
end



"""
    l2_penalty(m)
    
Sum of squared trainable parameters of model `m` 
"""
l2_penalty(m) = mapreduce(p -> sum(abs2, p), +, Flux.trainables(m); init=0.0)