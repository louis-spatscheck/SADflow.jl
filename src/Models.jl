# src/Models.jl
#
# Model architecture: the periodic-boundary convolution layer, the
# momentum-space "effective propagator" branch, and the two-branch
# transport network that combines them.

"""
    PeriodicConv(kernel, ch; stride=1, dilation=1, σ=identity, bias=true)

A `Flux.Conv` layer wrapped with circular (periodic) padding on all
spatial dims, so output size == input size under periodic BC.
`kernel` e.g. `(3,3)`, `ch` is `in => out`.
"""
struct PeriodicConv{C<:Conv,P}
    conv::C
    pads::P
end

Functors.@functor PeriodicConv (conv,)

function PeriodicConv(kernel::NTuple{N,Int}, ch::Pair{Int,Int};
                       stride=1, dilation=1, σ=identity, bias=true) where N
    pads = ntuple(N) do i
        k     = kernel[i]
        d     = dilation isa Tuple ? dilation[i] : dilation
        total = (k - 1) * d
        pl    = total ÷ 2
        pr    = total - pl
        (pl, pr)
    end
    conv = Conv(kernel, ch, σ; stride=stride, dilation=dilation, pad=0, bias=bias)
    return PeriodicConv(conv, pads)
end

(pc::PeriodicConv)(x::AbstractArray) = pc.conv(pad_periodic(x, pc.pads))

"""
    pool_field(x, pool_size)

Max-pool the first (time) dimension of `x` by `pool_size`. Not used by
`make_model`; kept as a utility for alternative conditioning schemes.
"""
function pool_field(x, pool_size)
    Nₜ, Nₓ, C, B = size(x)
    Nₜ_pool = Nₜ ÷ pool_size
    x_blocked = reshape(x, pool_size, Nₜ_pool, Nₓ, C, B)
    return dropdims(maximum(x_blocked, dims=1), dims=1)
end

"""
    EffectivePropagator(Nₜ; pool_size=4)

Small MLP `Σnet` (hidden width 16) mapping the lattice momenta `p̂₀` and
the ensemble moments `⟨φ²⟩`, `⟨φ⁴⟩` of each configuration to a
momentum-space self-energy `Σ(p)`, used to build an effective propagator
`1 / (p̂₀² + Σ(p))`.
"""
struct EffectivePropagator
    Σnet::Chain
    Nₜ::Int
end

Functors.@functor EffectivePropagator (Σnet,)

function EffectivePropagator(Nₜ::Int; pool_size=4)
    n_in = 2 + Nₜ
    Σnet = Flux.f64(Chain(
        Dense(n_in, 16, tanh),
        Dense(16, Nₜ, softplus),
    ))
    return EffectivePropagator(Σnet, Nₜ)
end

function (ep::EffectivePropagator)(p̂₀::AbstractVector, x::AbstractArray; pool_size=4)
    T = eltype(x)
    B = size(x, 4)

    φ² = mean(x .^ 2, dims=(1, 2, 3))[1, 1, 1, :]
    φ⁴ = mean(x .^ 4, dims=(1, 2, 3))[1, 1, 1, :]

    p̂₀T = T.(p̂₀)
    p̂₀B = repeat(p̂₀T, 1, B)

    cond = vcat(p̂₀B, φ²', φ⁴')
    Σ = ep.Σnet(cond)

    return @. inv(p̂₀B^2 + Σ)
end

"""
    stack_complex_flux(x)

Stack real/imag parts of a complex array along the channel dim (dim N-1).
"""
stack_complex_flux(x) = cat(real.(x), imag.(x), dims=ndims(x) - 1)

"""
    build_source_flux(x, vol)

Embed a `(T, 1, C, B)` field at the t-only slice back into full lattice
volume `vol = (T, spatial...)`, zero-padding the spatial dims.
"""
function build_source_flux(x, vol)
    (T, V...) = vol
    B = last(size(x))
    x = reshape(x, T, (V .÷ V)..., 1, B)
    o = zeros(eltype(x), T, (V .- 1)..., 1, B)
    return cat(x, o, dims=(1:length(V)) .+ 1)
end

"""
    ModelWrapper(m)

Thin wrapper so `trJ`/`trJJ` (Losses.jl) can treat the transport network
as a plain callable.
"""
struct ModelWrapper
    m
end
(w::ModelWrapper)(z) = w.m(z)

"""
    make_model(space, params; nodes=16, activation=tanh)

The transport network `f_θ` used in the thesis: the sum of

- a Fourier-space branch (FFT → 1×1 convolutions on (Re, Im) →
  `EffectivePropagator` → embed at zero spatial momentum → IFFT, scaled by
  `N_x / 2κ`), and
- a real-space branch of two `PeriodicConv` layers, `1 → nodes → 1`.
"""
function make_model(space::Grid, params::Phi4Params{T}; nodes=16, activation=tanh) where T

    vol = space.iL
    Nₜ = vol[1]
    Nₓ = prod(vol[2:end])

    
    factor = Nₓ / (T(2) * params.κ)

    p₀  = T(2π / Nₜ) .* T.(0:Nₜ-1)
    p̂₀  = T(2) .* sin.(p₀ ./ T(2))

    ep = EffectivePropagator(Nₜ)

    fourier_branch = Chain(
        x -> fft(x, 1:2),
        x -> stack_complex_flux(x),
        Chain(Conv((1, 1), 2 => 2, relu), Conv((1, 1), 2 => 1, identity)),
        x -> reshape(ep(p̂₀, x), Nₜ, 1, 1, size(x, 4)),
        fp -> build_source_flux(fp, vol),
        Fp -> real.(ifft(Fp, 1:2)) .* factor,
    ) |> Flux.f64

    conv_branch = Chain(
        PeriodicConv((3, 3), 1 => nodes; σ=activation, bias=false),
        PeriodicConv((3, 3), nodes => 1; σ=identity),
    ) |> Flux.f64

    return Chain(x -> fourier_branch(x) .+ conv_branch(x))
end

"""
    make_CNN(space, params; nodes=16, activation=tanh)

Alternative, purely real-space network: five `PeriodicConv` layers
`1 → nodes → … → nodes → 1` followed by `exp`. Not used in the thesis runs.
"""
function make_CNN(space::Grid, params::Phi4Params{T}; nodes=16, activation=tanh) where T

    vol = space.iL
    Nₜ = vol[1]
    Nₓ = prod(vol[2:end])

    conv_branch = Chain(
        PeriodicConv((3, 3), 1 => nodes; σ=activation, bias=false),
        PeriodicConv((3, 3), nodes => nodes; σ=activation, bias=false),
        PeriodicConv((3, 3), nodes => nodes; σ=activation, bias=false),
        PeriodicConv((3, 3), nodes => nodes; σ=activation, bias=false),
        PeriodicConv((3, 3), nodes => 1; σ=identity),
    ) |> Flux.f64

    return Chain(x -> conv_branch(x),
                x -> exp.(x))
end

const ACTIVATIONS = Dict(
    "tanh"     => tanh,
    "relu"     => relu,
    "sigmoid"  => Flux.σ,
    "swish"    => swish,
    "gelu"     => gelu,
    "softplus" => softplus,
    "identity" => identity,
)

"""
    activation_fn(name::String)

Look up an activation function by its command-line name (see `ACTIVATIONS`).
"""
function activation_fn(name::String)
    return get(ACTIVATIONS, name) do
        error("Unknown activation: $name. Valid: $(collect(keys(ACTIVATIONS)))")
    end
end