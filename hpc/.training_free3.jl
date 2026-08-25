using Pkg
Pkg.activate(".")
using ADerrors
using FormalSeries

using Plots
using ArgParse
using LinearAlgebra
using ForwardDiff
#using Lattice 

using Flux
using Flux.Losses: mse
using Functors
using Random
using Statistics
using Zygote

import BSON
import JLD2

import AbstractFFTs: fft, ifft
import ForwardDiff: Dual, partials, value, Partials
using DifferentiationInterface

function parse_commandline()
    args = ArgParseSettings()
    @add_arg_table args begin

        "--kappa", "-k"
            arg_type = Float64

        "--lambda", "-l"
            arg_type = Float64

        "--epochs", "-e"
            arg_type = Int
            default = 100

        "--batchsize", "-b"
            arg_type = Int 
            default = 64 
        
        "--test_batchsize"
            arg_type = Int
            default = 512
        
        "--depth", "-d"
            arg_type = Int  
            default = 1

        "--nodes", "-n"
            arg_type = Int  
            default = 1

        "--lr", "-r"
            arg_type = Float64
            default = 1e-3

        "--split"
            arg_type = Float64
            default = 0.75

        "-m", "--modes"
            arg_type = Int
            default = 1
        
        "-a", "--activation"
            arg_type = String
            default = "tanh"

        "--save_every"
            arg_type = Int
            default = 100

    end

    return parse_args(args)
end

#----------------UTILS ------------------

struct Phi4_params{T}
    κ::T
    λ::T
end

const BC_PERIODIC = 0
const BC_SF_ORBI  = 1
const BC_SF_AFWB  = 2
const BC_OPEN     = 3

struct Grid{N,M,B,D}
    ndim::Int64                           
    iL::NTuple{N,Int64}                   
    npls::Int64                          
    plidx::NTuple{M,Tuple{Int64, Int64}} 

    blk::NTuple{N,Int64}     
    blkS::NTuple{N,Int64} 
    rbk::NTuple{N,Int64} 
    rbkS::NTuple{N,Int64}

    bsz::Int64
    rsz::Int64

    ntw::NTuple{M,Int64}
    
    function Grid{N}(x, y, nt::Union{Nothing,NTuple{I,Int64}}=nothing) where {N,I}
        M = convert(Int64, round(N*(N-1)/2))
        N == length(x) || throw(ArgumentError("Lattice size incorrect length for dimension $N"))
        N == length(y) || throw(ArgumentError("Block   size incorrect length for dimension $N"))

        if any(i->i!=0, x.%y)
            error("Lattice size not divisible by block size.")
        end
        
        pls = Vector{Tuple{Int64, Int64}}()
        for i in N:-1:1
            for j in 1:i-1
                push!(pls, (i,j))
            end
        end

        r  = div.(x, y)
        rS = ones(N)
        yS = ones(N)
        for i in 2:N
            for j in 1:i-1
                rS[i] = rS[i]*r[j]
                yS[i] = yS[i]*y[j]
            end
        end

        D = prod(y)
        if nt == nothing
            ntw = ntuple(i->0, M)
        else
            ntw = nt
        end
        return new{N,M,BC_PERIODIC,D}(N, x, M, tuple(pls...), y,
                                    tuple(yS...), tuple(r...), tuple(rS...), prod(y), prod(r), ntw)
    end

    function Grid{N}(x, y, ibc::Int, nt::Union{Nothing,NTuple{I,Int64}}=nothing) where {N,I}
        M = convert(Int64, round(N*(N-1)/2))
        N == length(x) || throw(ArgumentError("Lattice size incorrect length for dimension $N"))
        N == length(y) || throw(ArgumentError("Block   size incorrect length for dimension $N"))

        if any(i->i!=0, x.%y)
            error("Lattice size not divisible by block size.")
        end

        if nt!=nothing
            if (ibc==BC_SF_AFWB) || (ibc==BC_SF_ORBI)
                if any(i->i!=0, nt[1:N-1])
                    error("Planes in T direction cannot be twisted with SF boundary conditions")
                end
            end
        end
        
        pls = Vector{Tuple{Int64, Int64}}()
        for i in N:-1:1
            for j in 1:i-1
                push!(pls, (i,j))
            end
        end

        r  = div.(x, y)
        rS = ones(N)
        yS = ones(N)
        for i in 2:N
            for j in 1:i-1
                rS[i] = rS[i]*r[j]
                yS[i] = yS[i]*y[j]
            end
        end

        D = prod(y)
        if nt == nothing
            ntw = ntuple(i->0,M)
        else
            ntw = nt
        end
        return new{N,M,ibc,D}(N, x, M, tuple(pls...), y,
                            tuple(yS...), tuple(r...), tuple(rS...), prod(y), prod(r), ntw)
    end
end

# ------------------------ Helpers ------------------------

function fft(x::AbstractArray{<:Dual{T,V,N}}, dims) where {T,V,N}
    vx = value.(x)
    px = ntuple(k -> partials.(x, k), N)
    fv = fft(vx, dims)
    fp = ntuple(k -> fft(px[k], dims), N)

    rd = similar(vx, Dual{T,V,N})
    id = similar(vx, Dual{T,V,N})
    for i in eachindex(vx)
        rd[i] = Dual{T,V,N}(real(fv[i]), Partials{N,V}(ntuple(k -> real(fp[k][i]), N)))
        id[i] = Dual{T,V,N}(imag(fv[i]), Partials{N,V}(ntuple(k -> imag(fp[k][i]), N)))
    end
    return complex.(rd, id)
end

function ifft(x::AbstractArray{<:Dual{T,V,N}}, dims) where {T,V,N}
    vx = value.(x)
    px = ntuple(k -> partials.(x, k), N)
    fv = ifft(vx, dims)
    fp = ntuple(k -> ifft(px[k], dims), N)

    rd = similar(vx, Dual{T,V,N})
    id = similar(vx, Dual{T,V,N})
    for i in eachindex(vx)
        rd[i] = Dual{T,V,N}(real(fv[i]), Partials{N,V}(ntuple(k -> real(fp[k][i]), N)))
        id[i] = Dual{T,V,N}(imag(fv[i]), Partials{N,V}(ntuple(k -> imag(fp[k][i]), N)))
    end
    return complex.(rd, id)
end

biject(z::AbstractArray{T,N}; dim=1) where {T,N} = begin
    n = size(z, dim) ÷ 2
    idx1 = ntuple(Returns(Colon()), dim-1)
    idx2 = ntuple(Returns(Colon()), N-dim)
    z[idx1..., 1:n, idx2...],
    z[idx1..., (n+1):end, idx2...]
end

dsum(x; dims) = dropdims(sum(x; dims=dims); dims=dims)

function shuffle_data(priors::AbstractArray)
    d = ndims(priors)
    N = size(priors, d)
    shuffled_indices = randperm(N)
    inds = ntuple(i -> (i == d ? shuffled_indices : :), d)
    return priors[inds...], inds
end

function select_random_batch(priors::AbstractArray, batch_size::Int)
    d = ndims(priors)
    if d < 1
        error("select_random_batch: priors must have at least 1 dimension")
    end
    N = size(priors, d)
    selected_indices = rand(1:N, batch_size)
    inds = ntuple(i -> (i == d ? selected_indices : :), d)
    return priors[inds...], inds
end

#------------- Loss Function ------------------

vjv(rng, func, z) = begin
    T = eltype(z)
    η = T.(rand(rng, [-1,1], size(z)...))
    Jη = ForwardDiff.derivative(t -> func(z .+ t .* η), zero(T))
    eachslice(η .* Jη, dims=ndims(η)) .|> sum
end

trJ(rng, func, z; ns=1) =
    [vjv(rng, func, z) for _ in 1:ns] |> stack

trJ(func, z; ns=1) =
    trJ(Random.default_rng(), func, z; ns=ns)

vjjv(rng, func, z) = begin
    T = eltype(z)
    η   = T.(rand(rng, [-1,1], size(z)...))
    Jη  = ForwardDiff.derivative(t -> func(z .+ t .* η),  zero(T))
    JJη = ForwardDiff.derivative(t -> func(z .+ t .* Jη), zero(T))
    eachslice(η .* JJη, dims=ndims(η)) .|> sum
end

trJJ(rng, func, z; ns=1) = begin
    raw = [vjjv(rng, func, z) for _ in 1:ns] |> stack
    mean(raw, dims=2)
end

trJJ(func, z; ns=1) = trJJ(Random.default_rng(), func, z; ns=ns)

vjv(rng, func, z) = begin
    T = eltype(z)
    η = T.(rand(rng, [-1,1], size(z)...))
    Jη = ForwardDiff.derivative(t -> func(z .+ t .* η), zero(T))
    eachslice(η .* Jη, dims=ndims(η)) .|> sum
end


trJ(rng, func, z; ns=1) =
    [vjv(rng, func, z) for _ in 1:ns] |> stack

trJ(func, z; ns=1) =
    trJ(Random.default_rng(), func, z; ns=ns)



vjjv(rng, func, z) = begin
    T = eltype(z)
    η   = T.(rand(rng, [-1,1], size(z)...))
    Jη  = ForwardDiff.derivative(t -> func(z .+ t .* η),  zero(T))   # J·η
    JJη = ForwardDiff.derivative(t -> func(z .+ t .* Jη), zero(T))   # J²·η
    eachslice(η .* JJη, dims=ndims(η)) .|> sum
end

trJJ(rng, func, z; ns=1) = begin
    raw = [vjjv(rng, func, z) for _ in 1:ns] |> stack
    mean(raw, dims=2)   # average over ns samples → (N,) per-sample estimates
end

trJJ(func, z; ns=1) = trJJ(Random.default_rng(), func, z; ns=ns)

function ∇f_sq(x::AbstractArray{Float64,Nd}, F; K=5) where Nd
    (X,Y,c,N) = size(x)
    tr2 = 0.0f0        
    f, back = Zygote.pullback(z -> F(z), x)
    for k in 1:K
        v = randn(Float64, size(x)) 
        Jv = back(v)[1]
        tr2 += sum(Jv .* Jv)
    end
    return f, tr2 / K / (N*c)
end

function ∇f_sq_exact(x::AbstractArray{Float64,4}, F)
    (X,Y,c,N) = size(x)
    f = F(x)
    tr2 = 0.0f0
    D = X * Y * c
    f_single = z -> vec(F(reshape(z, X, Y, c, 1))[:,:,:,1])
    for i in 1:N
        J = ForwardDiff.jacobian(f_single, vec(x[:,:,:,i]))
        tr2 += sum(J .^ 2)
    end
    return f, tr2 / (N*c)
end

function ∇f_sq_exact_linear(F)
    weight_matrix = F.layers[3].W 
    return real(sum(abs2, weight_matrix))
end

function f∇O(phi)
    return sum(phi[1, :, :, :],dims=(1,2))
end

function staple(f::AbstractArray{T,Nd}) where {T, Nd}

    fxp = circshift(f, (-1, 0, 0, 0))
    fyp = circshift(f, (0, -1, 0, 0))
    fxp_ = circshift(f, (1, 0, 0, 0))
    fyp_ = circshift(f, (0, 1, 0, 0))
    staple_sum = fxp .+ fyp .+ fxp_ .+ fyp_
    return staple_sum
end

function f_HS_f(phi, f, params)
    interaction = -2f0 * params.κ .* (f .* staple(f))
    prefac = (2f0 - 4f0 * params.λ) .+ 12f0 * params.λ .* (phi .^ 2)
    potential = prefac .* (f .^ 2)

    H_local = interaction .+ potential
    
    return sum(H_local, dims=(1,2,3))
end

struct ModelWrapper
    m
end

(w::ModelWrapper)(z) = w.m(z)

function KLloss(z::AbstractArray{Float64,Nd}, F ,σ², params; K=5, analytics=false) where Nd

    func = ModelWrapper(F)
    f_z = func(z)

    f1 = Zygote.ignore() do 
        mean(trJJ(func, z; ns=K))
    end 

    f2 = mean(f_HS_f(z, f_z, params))
    f3 = -2.0 * mean(f∇O(f_z))

    if analytics
        return 0.5f0 * (σ² + f1 + f2 + f3), f1, f2, f3
    end 
    
    return 0.5f0 * (σ² + f1 + f2 + f3)
end

function l2_penalty(m)
    return mapreduce(p -> sum(abs2, p), +, Flux.trainables(m); init=0.0)
end

# -------------------Metrics ------------------

function neighbour_sum(x)
    return circshift(x, (-1,0,0,0)) .+ circshift(x, (0,-1,0,0)) 
end

function ActionPhi4(x, kappa, lambda) 
    return sum(-2*kappa .* x .* neighbour_sum(x) .+ x.^2 .+ lambda .* (x.^2 .- 1.0).^2, dims=(1,2,3)) 
end

import Base: broadcastable
Base.broadcastable(s::uwreal) = Ref(s)

function ADerrors.uwreal(obs::Vector{FormalSeries.Series{T, N}},
                         ID::String,
                         rep=nothing
                         ) where {T, N}
    if rep == nothing
        rep = [length(obs)]
    end
    uwobs = FormalSeries.Series{ADerrors.uwreal, N}(
        ntuple(i -> ADerrors.uwreal([real(obs[j].c[i]) for j in 1:length(obs)], ID, rep), N)
    )
    return uwobs
end

function ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal, N},
                        wpm::Union{Dict{Int64,Vector{Float64}},Dict{String,Vector{Float64}}}
                        ) where N
    for i in 1:N
         ADerrors.uwerr(obs[i],wpm)
    end
end

ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal, N}) where N =
    ADerrors.uwerr(obs, Dict{String, Vector{Float64}}())

import Base.:+
Base.:+(a::uwreal) = a

uwassign(obs::Vector{Series{Float64, N}}, i; tagid="Ensemble") where {N} =
    uwreal([obs[k][i] for k in 1:length(obs)], tagid)

function neighbour_sum(x)
    return circshift(x, (-1,0)) .+ circshift(x, (0,-1)) 
end

function ActionPhi4(x, kappa, lambda) 
    return sum(-2*kappa .* x .* neighbour_sum(x) .+ x.^2 .+ lambda .* (x.^2 .- 1.0).^2, dims=(1,2)) 
end

function correlator(interpol_chain::Array{T,2}, tag::String, niter_vec::Vector{Int64};
                    wpm=Dict{String,Vector{Float64}}(), ed=100, timeav=false) where T

    L = length(interpol_chain[:,1])
    chainsize = length(interpol_chain[1,:])

    mean_uw = Vector{uwreal}(undef, L)

    for i in 1:L
        mean_uw[i] = uwreal(interpol_chain[i,:], tag, niter_vec)
    end

    dif_chain = similar(interpol_chain)
    for k in 1:chainsize
        dif_chain[:,k] .= interpol_chain[:,k] .- ADerrors.value.(mean_uw)
    end

    ct = Vector{uwreal}(undef, L)

    if !(timeav)
        for i in 1:L
            ct[i] = uwreal(dif_chain[i,:].*dif_chain[1,:], tag, niter_vec)
        end
    else
        for t in 1:L
            ct[t] = uwreal(0.0)

            for ti in 1:L
                tt = ti+t
                if ti+t > L
                    tt = ti+t-L
                end
                ct[t] += uwreal(dif_chain[tt,:].*dif_chain[ti,:]./L, tag, niter_vec)
            end
        end
        ct = reverse(ct)
    end

    return ct
end

# ---------------- Load Training Data ------------------

pargs = parse_commandline()

L1 = 32
L2 = 8
b = 8

kappa = pargs["kappa"]
lambda = pargs["lambda"]

max_epochs = pargs["epochs"]
bs_parse = pargs["batchsize"]

depth = pargs["depth"]
nodes = pargs["nodes"]

split = pargs["split"]
lr = pargs["lr"]

modes = pargs["modes"]
activ = pargs["activation"]
test_batchsize = pargs["test_batchsize"]
save_every = pargs["save_every"]

weight_decay = 0.0
eta = 1.0
start_decay = 20


N_eval = 1000
T = Float64

batchsize = Int(floor(bs_parse))
global lr

params_phi4 = Phi4_params(kappa, lambda)

space = Grid{2}((L1,L2),(b,b))
D = prod(space.iL)

JLD2.@load "./priors/2d_l$(lambda)_k$(kappa)_L_$(L1)_$(L2).jld2" pics 

phi = convert(Array{T,4}, pics[:,:,:,:20000:end])
pics = convert(Array{T,4}, pics[:,:,:,10000:20000])

N = size(pics,4)

maxiter = div(N, batchsize) - 1
iters = maxiter

x = vec(sum(phi[1, :, :, :], dims=(1)))
μ = mean(x)
var_z = mean((x .- μ).^2)

train_indices = randperm(N)[1:Int(floor(split * N))]
test_indices = setdiff(1:N, train_indices)

prior = pics[:, :, :, train_indices]
prior_test = pics[:, :, :, test_indices]

N_train = size(prior, 4)
N_test = size(prior_test, 4)
maxiter = div(N_train,batchsize)
iters = maxiter

mkpath("./models_free1")
mkpath("./data_free1")
mkpath("./plots_free1")


# ─────────────────────────────────────────────────────────────────────────────
# Periodic padding helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    pad_periodic(x, pads)

Pad spatial dims of `x` with periodic (circular) boundary conditions.
`x` has shape `(spatial..., C, B)`. `pads` is a tuple of pad widths for each
spatial dim — either an `Int` (symmetric pad) or a `Tuple{Int,Int}` (left, right).
"""
function pad_periodic(x::AbstractArray, pads)
    N = ndims(x) - 2                 # number of spatial dims
    @assert length(pads) == N "pads must match number of spatial dims"

    out = x
    for d in 1:N
        p = pads[d]
        (pl, pr) = p isa Tuple ? p : (p, p)
        pl == 0 && pr == 0 && continue

        sz   = size(out, d)
        left  = selectdim(out, d, (sz - pl + 1):sz)
        right = selectdim(out, d, 1:pr)
        out   = cat(left, out, right; dims=d)
    end
    return out
end

# ─────────────────────────────────────────────────────────────────────────────
# PeriodicConv layer
# ─────────────────────────────────────────────────────────────────────────────

struct PeriodicConv{C<:Conv,P}
    conv::C
    pads::P
end

Functors.@functor PeriodicConv (conv,)

"""
    PeriodicConv(kernel, ch; stride=1, dilation=1, σ=identity, bias=true)

A `Conv` layer with periodic (circular) boundary conditions on all spatial dims.
`kernel` is a tuple like `(3, 3)`, `ch` is `in => out`.
"""
function PeriodicConv(kernel::NTuple{N,Int}, ch::Pair{Int,Int};
                      stride=1, dilation=1, σ=identity, bias=true) where N

    # Padding needed for "same" output under periodic BC:
    # for kernel k and dilation d, total pad = (k-1)*d, split left/right.
    pads = ntuple(N) do i
        k = kernel[i]
        d = dilation isa Tuple ? dilation[i] : dilation
        total = (k - 1) * d
        pl = total ÷ 2
        pr = total - pl
        (pl, pr)
    end

    conv = Conv(kernel, ch, σ; stride=stride, dilation=dilation, pad=0, bias=bias)
    return PeriodicConv(conv, pads)
end

function (pc::PeriodicConv)(x::AbstractArray)
    return pc.conv(pad_periodic(x, pc.pads))
end

function pool_field(x, pool_size)
    Nₜ, Nₓ, C, B = size(x)
    Nₜ_pool = Nₜ ÷ pool_size

    x_blocked = reshape(x, pool_size, Nₜ_pool, Nₓ, C, B)
    return dropdims(maximum(x_blocked, dims=1), dims=1)
end

struct EffectivePropagator
    Σnet::Chain
    Nₜ::Int
end

Functors.@functor EffectivePropagator (Σnet,)

function EffectivePropagator(Nₜ::Int;nodes=16, pool_size=4)

    Nₜ_pool = Nₜ ÷ pool_size
    n_in = 2 + Nₜ

    Σnet = Flux.f64(Chain(
        Dense(n_in, 16, tanh),
        #Dense(nodes, nodes, activ),
        Dense(16, Nₜ, softplus)
    ))

    #Znet = Flux.f64(Chain(
    ##    Dense(n_in, nodes, activ),
    #    #Dense(nodes,nodes, activ),
    #    Dense(nodes, Nₜ, identity)
    #))

    return EffectivePropagator(Σnet, Nₜ)
end

function (ep::EffectivePropagator)(p̂₀::AbstractVector, x::AbstractArray; pool_size=4)

    T = eltype(x)
    B = size(x, 4)

    φ²  = mean(x.^2, dims=(1,2,3))[1,1,1,:]
    φ⁴  = mean(x.^4, dims=(1,2,3))[1,1,1,:]

    #x_pool = pool_field(x, pool_size)
    #x_pool = mean(x_pool, dims=2)[:,1,1,:]

    p̂₀T = T.(p̂₀)
    p̂₀B = repeat(p̂₀T, 1, B)

    cond = vcat(p̂₀B, φ²', φ⁴')

    Σ = ep.Σnet(cond)
    #Z = ep.Znet(cond)

    fp = @.  inv(p̂₀B^2 + Σ)

    return fp
end

function stack_complex_flux(x)
    cat(real.(x), imag.(x), dims=ndims(x)-1)
end

function build_source_flux(x, vol)

    (T, V...) = vol
    B = last(size(x))

    x = reshape(x, T, (V .÷ V)..., 1, B)
    o = zeros(eltype(x), T, (V .- 1)..., 1, B)

    return cat(x, o, dims=(1:length(V)) .+ 1)
end
##
# ─────────────────────────────────────────────────────────────────────────────
# Named layers replacing anonymous functions
# ─────────────────────────────────────────────────────────────────────────────

struct FFTLayer
    dims::UnitRange{Int}
end
Functors.@functor FFTLayer ()
(l::FFTLayer)(x) = fft(x, l.dims)

struct StackComplex end
Functors.@functor StackComplex ()
(::StackComplex)(x) = stack_complex_flux(x)

struct EPApply{T}
    ep::EffectivePropagator
    p̂₀::Vector{T}
    Nₜ::Int
end
Functors.@functor EPApply (ep,)
(l::EPApply)(x) = reshape(l.ep(l.p̂₀, x), l.Nₜ, 1, 1, size(x, 4))

struct BuildSource{V}
    vol::V
end
Functors.@functor BuildSource ()
(l::BuildSource)(fp) = build_source_flux(fp, l.vol)

struct IFFTLayer{T}
    dims::UnitRange{Int}
    factor::T
end
Functors.@functor IFFTLayer ()
(l::IFFTLayer)(Fp) = real.(ifft(Fp, l.dims)) .* l.factor

struct TwoBranch{A,B}
    fourier::A
    conv::B
end
Functors.@functor TwoBranch
(m::TwoBranch)(x) = m.fourier(x) .+ m.conv(x)

# ─────────────────────────────────────────────────────────────────────────────
# Refactored make_model1 — identical behavior, no closures
# ─────────────────────────────────────────────────────────────────────────────

function make_model1(vol, κ::T; depth=2, nodes=4, activ=relu) where T
    Nₜ = first(vol)
    Nₓ = prod(vol[2:end])
    factor = T(Nₓ / 2κ)

    p₀ = T(2π / Nₜ) .* T.(0:Nₜ-1)
    p̂₀ = T(2) .* sin.(p₀ ./ T(2))

    ep = EffectivePropagator(Nₜ)

    fourier_branch = Chain(
        FFTLayer(1:2),
        StackComplex(),
        Chain(Conv((1,1), 2 => 2, relu), Conv((1,1), 2 => 1, identity)),
        EPApply(ep, p̂₀, Nₜ),
        BuildSource(vol),
        IFFTLayer(1:2, factor),
    ) |> Flux.f64

    conv_branch = Chain(
        PeriodicConv((3,3), 1 => nodes; σ=activ, bias=false),
        PeriodicConv((3,3), nodes => nodes; σ=activ, bias=false),
        PeriodicConv((3,3), nodes => 1; σ=identity),
    ) |> Flux.f64

    return TwoBranch(fourier_branch, conv_branch)
end
const ACTIVATIONS = Dict(
    "tanh"     => tanh,
    "relu"     => relu,
    "sigmoid"  => σ,
    "swish"    => swish,
    "gelu"     => gelu,
    "softplus" => softplus,
    "identity" => identity,
)

activ_fn = get(ACTIVATIONS, activ) do
    error("Unknown activation: $activ. Valid: $(collect(keys(ACTIVATIONS)))")
end

model = make_model1((L1, L2), kappa; nodes=nodes, activ=activ_fn)

function KLloss_batch(z, F, σ², params; K=5)

    func = ModelWrapper(F)
    f_z = func(z)

    f1 = Zygote.ignore() do 
        mean(trJJ(func, z; ns=K),dims=2)
    end

    f2 = f_HS_f(z, f_z, params) 
    f3 = -2.0 * f∇O(f_z)

    return 0.5 * mean(f1 .+ f2 .+ f3 .+ σ²)
end

function compute_force(ϕ::Array{T,4}, params) where {T}
    nbr_sum = circshift(ϕ, (-1, 0, 0, 0)) .+ circshift(ϕ, ( 1, 0, 0, 0)) .+
              circshift(ϕ, ( 0,-1, 0, 0)) .+ circshift(ϕ, ( 0, 1, 0, 0))

    return @. T(2)*ϕ + T(4)*params.λ * (ϕ^2 - one(T)) * ϕ - T(2)*params.κ * nbr_sum
end


function metrics(func,ϕ, params, var_z, mean_z;ns=5)

    f = func(ϕ)

    #traceJ = mean(trJ(func, ϕ; ns), dims=2)[:,1]
    traceJJ = trJJ(func, ϕ; ns)[:,1]
    #force = compute_force(ϕ,params)
    hessian = f_HS_f(ϕ, f, params)[1,1,1,:]

    #source_phi_var = (sum(ϕ[1:1, :, :, :],dims=(2)) .- mean_z)[1,1,1,:]
    sourcef = sum(f[1:1, :, :, :],dims=(2))[1,1,1,:]

    #stein = traceJ .- sum(f .* force, dims=(1,2,3))[1,1,1,:]


    #loss = traceJJ .+ hessian .- 2.0 .* sourcef .+ var_z

    #stein_loss = (stein .- source_phi_var).^2

    return traceJJ, hessian, sourcef#, stein, loss, stein_loss
end

function plot_training(train_loss, test_loss, ess, N;
                       title="Training Overview")

    epochs = 1:10:N

    p = plot(
        epochs, abs.(train_loss[epochs] .- minimum(train_loss[1:N]) * 1.1),
        label       = "Train loss",
        color       = "#2166ac",
        linewidth   = 2.5,
        xlabel      = "Epoch",
        ylabel      = "Loss",
        title       = title,
        yscale      = :log10,
        titlefont   = font(13, "Computer Modern"),
        tickfont    = font(10, "Computer Modern"),
        guidefont   = font(11, "Computer Modern"),
        legendfont  = font(9,  "Computer Modern"),
        legend      = :topright,
        grid        = true,
        gridalpha   = 0.3,
        gridstyle   = :dash,
        framestyle  = :box,
        size        = (900, 450),
        dpi         = 150,
        margin      = 5Plots.mm,
        top_margin  = 6Plots.mm,
    )

    plot!(p, epochs, abs.(test_loss[epochs] .- minimum(test_loss[1:N]) * 1.1),
        label     = "Test loss",
        color     = "#d6604d",
        linewidth = 2.5,
    )

    p2 = twinx(p)
    ess_idx = findall(>(0), ess[1:N])


    plot!(p2, ess_idx, 1 ./ ess[ess_idx],
        label      = "ESS",
        color      = "#4dac26",
        linewidth  = 2.5,
        yscale     = :log10,
        linestyle  = :dash,
        ylabel     = "ESS",
        guidefont  = font(11, "Computer Modern"),
        tickfont   = font(10, "Computer Modern"),
        legendfont = font(9,  "Computer Modern"),
        legend     = :bottomright,
        grid       = false,
    )

    return p
end

loss_function = KLloss_batch

fval = zeros(Float64, max_epochs)
ftest = zeros(Float64, max_epochs)
wESS = zeros(Float64, max_epochs)

traceJJS = zeros(Float64, max_epochs)
hessians = zeros(Float64, max_epochs)
sourcefs = zeros(Float64, max_epochs)


optimiserT(lr) = OptimiserChain(
    WeightDecay(weight_decay),
    Flux.Adam(lr)
)

opt = Flux.setup(optimiserT(lr), model)

for epoch in 1:max_epochs

    shuffled_prior = shuffle_data(prior)[1]
    loss_sum = 0.0f0

    for j in 1:1

        start_idx = Int((j-1)*batchsize) + 1
        end_idx = Int(j*batchsize)

        Xbatch = shuffled_prior[:,:,:, start_idx:end_idx]

        loss_val, grads = Flux.withgradient(model) do ml
            loss_function(Xbatch, ml, var_z, params_phi4; K=5)
        end

        Flux.update!(opt, model, grads[1])


    end
    if epoch % 10 == 1
        X_batch = select_random_batch(prior, test_batchsize)[1]
        loss_train = loss_function(X_batch, model, var_z, params_phi4; K=10)
        fval[epoch] = loss_train

        X_test_batch = select_random_batch(prior_test, test_batchsize)[1]
        loss_test = loss_function(X_test_batch, model, var_z, params_phi4; K=10)

        ftest[epoch] = loss_test
        println("Epoch $epoch | Train Loss: $(fval[epoch])")
        println("Epoch $epoch | Test Loss: $(ftest[epoch])")
    end

    if epoch % 50 == 1

        ϵ = Series{Float64,2}((0.f0,1.f0))

        f_model = model(prior_test[:,:,:,1:N_eval])

        PHI = Array{Series{T,2},3}(undef, L1, L2 , N_eval)

        for n in 1:N_eval
            for i in 1:L1
                for j in 1:L2
                    PHI[i,j,n] = Series{T,2}((prior_test[i,j,1,n], 0.0))
                end
            end
        end

        PHIT = Array{Series{Float64,2},3}(undef, L1, L2 , N_eval)

        for n in 1:N_eval
            for i in 1:L1
                for j in 1:L2
                    PHIT[i,j,n] = Series{Float64,2}((prior_test[i,j,1,n], f_model[i,j,1,n]))
                end
            end
        end

        PHIT_T = sum(PHIT, dims=2)[:,1,:]

        action(x) = ActionPhi4(x,kappa,lambda)[:]

        action_term = action(PHI) .- action(PHIT)

        source_term = sum(PHIT[1,:,:], dims=1)[1,:]

        jacobian_term = mean(trJ(model, prior_test[:,:,:,1:N_eval]; ns=10), dims=2)[:]

        tag = string(epoch)

        WT = exp.(action_term .+ ϵ .* source_term .+ ϵ .* jacobian_term)

        meanWT = uwreal(WT, tag)

        uwerr(meanWT)

        ess = ADerrors.err.(meanWT.c[2])

        wESS[epoch] = ess

        println("Epoch $epoch | Test ESS: $(ess)")

    end

    if epoch % save_every == 0 || epoch == max_epochs

        a = metrics(model, prior_test[:,:,:,1:test_batchsize], params_phi4, var_z, μ; ns=10)

        traceJJS[epoch] = mean(a[1])
        hessians[epoch] = mean(a[2])
        sourcefs[epoch] = mean(a[3])

        println("Epoch $epoch | traceJJ: $(traceJJS[epoch])")
        println("Epoch $epoch | hessian: $(hessians[epoch])")
        println("Epoch $epoch | sourcef: $(sourcefs[epoch])")

        BSON.@save "./models_free1/2d_l$(lambda)_k$(kappa)_L_$(L1)_$(L2)_depth$(depth)_nodes$(nodes)_bs$(bs_parse)_active$(activ)_lr$(lr)_e$(epoch).bson" model

        JLD2.@save "./data_free1/2d_l$(lambda)_k$(kappa)_L_$(L1)_$(L2)_depth$(depth)_nodes$(nodes)_bs$(bs_parse)_active$(activ)_lr$(lr).jld2" fval ftest wESS traceJJS hessians sourcefs

        p = plot_training(fval, ftest, wESS, epoch)

        savefig(
            p,
            "./plots_free1/2d_l$(lambda)_k$(kappa)_L_$(L1)_$(L2)_depth$(depth)_nodes$(nodes)_bs$(bs_parse)_active$(activ)_lr$(lr).png"
        )
    end
end




