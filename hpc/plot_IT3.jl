#/usr/bin/enva ‘julia’: No sucis_plots.jl
#
# Build the full set of analysis plots for a single (λ, κ) trained-flow run:
#   - training metrics (loss, wESS, F1/F2/F3 components)
#   - relative-error-vs-free-theory across epochs (when λ = 0)
#   - reweighted correlator and effective mass at selected epochs
#   - signal-to-noise at fixed t across epochs
#
# Usage:
#   julia generate_thesis_plots.jl --lambda 0.0 --kappa 0.2485 \
#        --L1 32 --L2 8 --depth 1 --nodes 4 --bs 256 --activ relu \
#        --lr 1e-4 --N 22000 --ncfg 10000 \
#        --epochs 1000,1800,10600,22200 \
#        --outdir ./results/figs_thesis
# =============================================================================
using Pkg
Pkg.activate(".")
using Dates
using ArgParse
using FileIO
using Statistics, LinearAlgebra
using Plots, LaTeXStrings
using ADerrors
using Flux, Functors                       # needed by the model code
using FFTW          
import JLD2
import BSON
using Random
import AbstractFFTs: fft, ifft
import ForwardDiff: Dual, partials, value, Partials
using Roots
using FormalSeries
using ForwardDiff
# -----------------------------------------------------------------------------

# TODO: confirm the actual paths. The file you pasted defines:
#   pad_periodic, PeriodicConv, EffectivePropagator, EPApply, FFTLayer,
#   StackComplex, BuildSource, IFFTLayer, TwoBranch, make_model1,
#   stack_complex_flux, build_source_flux, pool_field
# Pick whichever file(s) contain these:
#include(joinpath(@__DIR__, "..", "src", "model.jl"))
#include(joinpath(@__DIR__, "..", "src", "reweighting.jl"))   # reweighted_correlator, action, source
#include(joinpath(@__DIR__, "..", "src", "utils.jl"))         # nonzero_idx, dsum, sumvol, uwc, ...

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

function parse_commandline()
    s = ArgParseSettings(
        description   = "Generate the full plot set for a single (λ, κ) trained-flow run.",
        autofix_names = true,
    )

    @add_arg_table! s begin
        # --- Physical / lattice parameters
        "--lambda"
            arg_type = Float64; required = false
            help     = "φ⁴ coupling λ"
        "--kappa"
            arg_type = Float64; required = false
            help     = "hopping parameter κ"
        "--L1"
            arg_type = Int; default = 32
        "--L2"
            arg_type = Int; default = 8

        # --- Network / training hyperparameters (used in filenames)
        "--depth"
            arg_type = Int; default = 1
        "--nodes"
            arg_type = Int; default = 4
        "--bs"
            arg_type = Int; default = 256
        "--activ"
            arg_type = String; default = "relu"
        "--lr"
            arg_type = Float64; default = 1e-4

        # --- Analysis parameters
        "--N"
            arg_type = Int; default = 22000
            help     = "max epoch for training-metric plots (used by nonzero_idx)"
        "--ncfg"
            arg_type = Int; default = 10000
            help     = "number of configurations to use for correlator measurements"
        "--thermalize"
            arg_type = Int; default = 10000
            help     = "number of initial configurations to discard (thermalization)"

        "--Neval"
            arg_type = Int; default = 5000
            help     = "number of configurations to use for per-epoch correlator plots (must be ≤ ncfg)"
        "--epochs_load"
            arg_type = String; default = "200:400:22000"
            help     = "range/list of epochs to load model checkpoints from (Julia range or comma list)"
        "--epochs_show"
            arg_type = String; default = "1000,5000,10200,21000"
            help     = "epochs to compare in per-epoch plots (comma list)"
        "--snr_t"
            arg_type = Int; default = 2
            help     = "lattice time index for the SNR-vs-epoch plot (1-based)"

        # --- Paths
        "--datadir"
            arg_type = String; default = "./data_free1"
            help     = "directory containing the training-metrics .jld2 files"
        "--modeldir"
            arg_type = String; default = "./models_free1"
            help     = "directory containing the per-epoch .bson model files"
        "--cfgdir"
            arg_type = String; default = "./priors_plot"
            help     = "directory containing the field-configuration .jld2 files"
        "--outdir"
            arg_type = String; default = "./figs_thesis"
    end

    return parse_args(s)
end

# -----------------------------------------------------------------------------
# Filename / tag construction
# -----------------------------------------------------------------------------




# ─────────────────────────────────────────────────────────────────────────────
# Custom FFT methods for ForwardDiff
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# uwreal / Series glue
# ─────────────────────────────────────────────────────────────────────────────

import Base: broadcastable
Base.broadcastable(s::uwreal) = Ref(s)

function ADerrors.uwreal(obs::Vector{FormalSeries.Series{T, N}},
                         ID::String, rep=nothing) where {T, N}
    if rep === nothing
        rep = [length(obs)]
    end
    uwobs = FormalSeries.Series{ADerrors.uwreal, N}(
        ntuple(i -> ADerrors.uwreal([real(obs[j].c[i]) for j in 1:length(obs)], ID, rep), N)
    )
    return uwobs
end

function ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal, N},
                        wpm::Union{Dict{Int64,Vector{Float64}}, Dict{String,Vector{Float64}}}) where N
    for i in 1:N
        ADerrors.uwerr(obs[i], wpm)
    end
end

ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal, N}) where N =
    ADerrors.uwerr(obs, Dict{String, Vector{Float64}}())

import Base.:+
Base.:+(a::uwreal) = a

# ─────────────────────────────────────────────────────────────────────────────
# Parameter struct
# ─────────────────────────────────────────────────────────────────────────────

struct Phi4_params{T}
    κ::T
    λ::T
end

# Allow `params.kappa` / `params.lambda` access for ASCII compatibility
Base.getproperty(p::Phi4_params, name::Symbol) =
    name === :kappa  ? getfield(p, :κ) :
    name === :lambda ? getfield(p, :λ) :
    getfield(p, name)

# ─────────────────────────────────────────────────────────────────────────────
# Action and helpers (2D only — used for plotting/reweighting)
# ─────────────────────────────────────────────────────────────────────────────

neighbour_sum_2d(x) = circshift(x, (-1, 0)) .+ circshift(x, (0, -1))

ActionPhi4(x, κ, λ) =
    sum(-2 * κ .* x .* neighbour_sum_2d(x) .+ x.^2 .+ λ .* (x.^2 .- 1.0).^2, dims=(1,2))

function shuffle_data(priors::AbstractArray)
    d = ndims(priors)
    N = size(priors, d)
    idx = randperm(N)
    sel = ntuple(i -> (i == d ? idx : :), d)
    return priors[sel...], idx
end


"Common stem used for both training-metrics .jld2 and per-epoch .bson files."
function run_stem(p)
    "2d_l$(p.lambda)_k$(p.kappa)_L_$(p.L1)_$(p.L2)" *
    "_depth$(p.depth)_nodes$(p.nodes)_bs$(p.bs)" *
    "_active$(p.activ)_lr$(p.lr)"
end

"ADerrors ensemble name; depends only on the physical ensemble."
ensemble_tag(p) = "2d_l$(p.lambda)_k$(p.kappa)_L_$(p.L1)_$(p.L2)"

# Parse "200:400:22000" or "200,1000,1800" into a Vector{Int}
function parse_epoch_spec(spec::AbstractString)
    if occursin(':', spec)
        parts = parse.(Int, split(spec, ':'))
        length(parts) == 2 && return collect(parts[1]:parts[2])
        length(parts) == 3 && return collect(parts[1]:parts[2]:parts[3])
        error("Bad range spec: $spec")
    else
        return parse.(Int, strip.(split(spec, ',')))
    end
end

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

"""
    load_run(p, args) -> NamedTuple

Load the training-metrics file (fval, ftest, wESS, traceJJS, hessians, sourcefs).
"""
function load_run(p, args)
    fname = joinpath(args["datadir"], run_stem(p) * ".jld2")
    isfile(fname) || error("Training-metrics file not found: $fname")
    @info "Loading training metrics" file=fname

    JLD2.@load fname fval ftest wESS traceJJS hessians sourcefs
    return (fval=fval, ftest=ftest, wESS=wESS,
            traceJJS=traceJJS, hessians=hessians, sourcefs=sourcefs)
end



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


"""
    reconstruct_model(loaded, vol, κ) -> callable

Rebuild a working callable from a BSON-loaded `TwoBranch` whose
`fourier_branch` contains anonymous functions that didn't survive
serialization. Splices the trained `EffectivePropagator` and inner Conv chain
into a fresh wrapping `Chain`.
"""
function reconstruct_model(loaded, vol, κ)
    inner       = loaded.layers[1]
    conv_branch = inner.conv_branch
    loaded_fb   = inner.fourier_branch

    inner_chain = nothing
    eff_prop    = nothing
    for layer in loaded_fb.layers
        if layer isa Chain
            inner_chain = layer
        elseif layer isa EffectivePropagator
            eff_prop = layer
        end
    end
    @assert inner_chain !== nothing "Could not find inner Conv chain in loaded model"
    @assert eff_prop    !== nothing "Could not find EffectivePropagator in loaded model"

    Nₜ     = first(vol)
    Nₓ     = prod(vol[2:end])
    T_     = typeof(κ)
    factor = T_(Nₓ / 2κ)
    p₀     = T_(2π / Nₜ) .* T_.(0:Nₜ-1)
    p̂₀     = T_(2) .* sin.(p₀ ./ T_(2))

    fourier_branch_fresh = Chain(
        x  -> fft(x, 1:2),
        x  -> stack_complex_flux(x),
        inner_chain,
        x  -> reshape(eff_prop(p̂₀, x), Nₜ, 1, 1, size(x, 4)),
        fp -> build_source_flux(fp, vol),
        Fp -> real.(ifft(Fp, 1:2)) .* factor,
    ) |> Flux.f64

    return x -> fourier_branch_fresh(x) .+ conv_branch(x)
end

"""
    load_models(p, args, epochs_to_load) -> Dict{Int, callable}

Load per-epoch BSON checkpoints and wrap each through `reconstruct_model`
so the returned dict maps epoch → working callable `model(x)`.
"""
function load_models(p, args, epochs_to_load::Vector{Int}, model)
    vol = (p.L1, p.L2)
    stem = run_stem(p)
    models = Dict{Int, Any}()
    for e in epochs_to_load
        fname = joinpath(args["modeldir"], stem * "_e$(e).bson")
        if !isfile(fname)
            @warn "Missing checkpoint" file=fname
            continue
        end
        local model    # name BSON.@load expects to bind to
        BSON.@load fname model
        models[e] = model
        @info "Loaded checkpoint" epoch=e
    end
    isempty(models) && error("No model checkpoints could be loaded from $(args["modeldir"])")
    return models
end

"""
    load_configurations(args, p) -> Array{Float64, 4}

Load `pics`, shape (L1, L2, 1, N_cfg), of φ-field configurations for the
physical ensemble (depends on λ, κ, L1, L2).

TODO: confirm the actual filename and the variable name inside the .jld2.
"""
function load_configurations(args, p)
    fname = joinpath(args["cfgdir"], ensemble_tag(p) * ".jld2")
    isfile(fname) || error("Configurations file not found: $fname")
    @info "Loading configurations" file=fname

    JLD2.@load fname pics
    @assert ndims(pics) == 4 "Expected pics of shape (L1,L2,1,N); got $(size(pics))"
    @info "Loaded configurations" size=size(pics)
    return pics
end
#####################################################


# -----------------------------------------------------------------------------
# Plot: model output statistics (mean & variance) across epochs
# -----------------------------------------------------------------------------

"""
    stack_field_stats(models_dict, data; epochs, p, args,
                      logscale=true, shared_clims=true, saveas=nothing)

Stacked grid of heatmaps showing the per-site mean and variance of
`models_dict[e](data)` across the configuration axis, one row per epoch in
`epochs`. Two columns: mean (diverging colormap, signed-log) and variance
(sequential colormap, log10).
"""
function stack_field_stats(models_dict, data;
                           epochs,
                           p,
                           xlabel        = "x",
                           ylabel        = "t",
                           logscale      = true,
                           shared_clims  = true,
                           saveas        = nothing)

    n_rows = length(epochs)
    n_cfgs = size(data, ndims(data))

    # --- Pass 1: compute all stats
    stats = map(epochs) do ep
        f = models_dict[ep](data)
        configs_dim = Tuple(3:ndims(f))
        μ  = dropdims(mean(f; dims=configs_dim); dims=configs_dim)
        σ² = dropdims(var(f;  dims=configs_dim); dims=configs_dim)
        μ  = dropdims(μ;  dims=Tuple(d for d in 1:ndims(μ)  if size(μ, d)  == 1))
        σ² = dropdims(σ²; dims=Tuple(d for d in 1:ndims(σ²) if size(σ², d) == 1))
        (epoch = ep, μ = μ, σ² = σ²)
    end

    # --- Transforms
    function transform_mean(μ)
        if logscale
            ref = max(maximum(abs, μ), eps())
            sign.(μ) .* log10.(1 .+ abs.(μ) ./ ref .* 10)
        else
            μ
        end
    end
    transform_var(σ²) = logscale ? log10.(max.(σ², eps())) : σ²

    μ_panels  = [transform_mean(s.μ)  for s in stats]
    σ²_panels = [transform_var(s.σ²)  for s in stats]

    # --- Shared color limits
    if shared_clims
        cmax_μ = maximum(maximum(abs, m) for m in μ_panels)
        clims_μ = (-cmax_μ, cmax_μ)
        cmin_σ = minimum(minimum, σ²_panels)
        cmax_σ = maximum(maximum, σ²_panels)
        clims_σ = (cmin_σ, cmax_σ)
    else
        clims_μ = :auto
        clims_σ = :auto
    end

    # --- Build subplots
    plots = Plots.Plot[]
    for (i, s) in enumerate(stats)
        is_top    = (i == 1)
        is_bottom = (i == n_rows)

        push!(plots, heatmap(μ_panels[i]';
            title      = is_top ? "Mean ⟨f⟩" : "",
            xlabel     = is_bottom ? xlabel : "",
            ylabel     = "epoch $(s.epoch)\n$ylabel",
            color      = :balance,
            clims      = clims_μ,
            colorbar   = false,
            framestyle = :box,
        ))
        push!(plots, heatmap(σ²_panels[i]';
            title      = is_top ? "Variance Log(Var(f))" : "",
            xlabel     = is_bottom ? xlabel : "",
            ylabel     = "",
            color      = :viridis,
            clims      = clims_σ,
            colorbar   = is_top,
            framestyle = :box,
        ))
    end

    fig = plot(plots...;
        layout              = (n_rows, 2),
        size                = (1300, 340 * n_rows),
        dpi                 = 200,
        plot_title          = "Flow output across training " *
                              L"(\lambda=%$(p.lambda),\ \kappa=%$(p.kappa),\ N=%$(n_cfgs))",
        plot_titlefontsize  = 11,
        link                = :both,
    )

    if saveas !== nothing
        base, _ = splitext(saveas)
        savefig(fig, base * ".pdf")
    end

    return fig
end

# -----------------------------------------------------------------------------
# Driver helper — picks the right epochs and output path from args/p
# -----------------------------------------------------------------------------

"""
    plot_field_stats(models, pics, p, args; ncfg=nothing)

Wraps `stack_field_stats` with the conventions used by the rest of the script:
takes data from the first `ncfg` configurations, uses `epochs_show` from args,
and writes to `<outdir>/f_stats_<run_stem>`.
"""
function plot_field_stats(models, pics, p, args;
                          ncfg = nothing,
                          epochs_show = nothing)

    ncfg        = something(ncfg, args["ncfg"])
    epochs_show = something(epochs_show, parse_epoch_spec(args["epochs_show"]))

    # Filter to epochs actually present in the loaded models
    available = filter(e -> haskey(models, e), epochs_show)
    if length(available) != length(epochs_show)
        missing = setdiff(epochs_show, available)
        @warn "Skipping epochs not present in loaded models" missing
    end
    isempty(available) && error("None of epochs_show are present in the loaded models")

    data = @view pics[:, :, :, 1:ncfg]
    saveas = joinpath(args["outdir"], "f_stats_" * run_stem(p) * ".png")

    @info "Plotting field statistics" epochs=available ncfg=ncfg out=saveas

    return stack_field_stats(models, data;
                             epochs = available,
                             p      = p,
                             saveas = saveas)
end


# -----------------------------------------------------------------------------
# Reweighted correlator + per-epoch comparison plot
# -----------------------------------------------------------------------------

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


function reweighted_correlator(model, prior, params; N=2000, tag="rw", T=Float64, trace = true)
    L1, L2 = size(prior, 1), size(prior, 2)
    @assert size(prior, 4) ≥ N "Need at least $N configurations, got $(size(prior, 4))"

    
    ϕ_test   = prior[:, :, :, 1:N]
    f_model  = model(ϕ_test)

    PHI  = Array{Series{Float64, 2}, 3}(undef, L1, L2, N)
    PHIT = Array{Series{Float64, 2}, 3}(undef, L1, L2, N)
    println("Constructing PHI and PHIT series arrays...")
    for n in 1:N, i in 1:L1, j in 1:L2
        PHI[i, j, n]  = Series{Float64, 2}((ϕ_test[i, j, 1, n], zero(T)))
        PHIT[i, j, n] = Series{Float64, 2}((ϕ_test[i, j, 1, n], f_model[i, j, 1, n]))
    end

    println(size(PHIT))

    PHIT_T = sum(PHIT, dims=2)[:, 1, :]
    println("this takes long")

    ϵ  = Series{T, 2}((zero(Float64), one(Float64)))
    action_term = (ActionPhi4(PHI,  params.kappa, params.lambda)[:] .-
                  ActionPhi4(PHIT, params.kappa, params.lambda)[:])
    source_term = sum(PHIT[1, :, :], dims=1)[1, :]



    if trace
        trace_term = mean(trJ(model, ϕ_test ; ns=25), dims=2)[:]
        WT = exp.(action_term .+ ϵ .* source_term .+ ϵ .* trace_term)
    else
        WT = exp.(action_term .+ ϵ .* source_term)
    end

    mean_phi_t_trw = Vector{Series{uwreal, 2}}(undef, L1)
    for t in 1:L1
        mean_phi_t_trw[t] = uwreal(WT .* PHIT_T[t, :], tag) / uwreal(WT, tag)
    end
    println("actually")
    corr_trw  = [mean_phi_t_trw[t].c[2] for t in 1:L1]
    corr_norm = corr_trw ./ corr_trw[1]
    uwerr.(corr_norm)

    return (corr=corr_norm,WT=WT)
end
"""
    PLOT_KWARGS

Shared plotting defaults across all thesis figures.
"""
const PLOT_KWARGS = (
    grid           = true,
    gridalpha      = 0.2,
    minorgrid      = false,
    minorgridalpha = 0.1,
    tickfontsize   = 28,
    guidefontsize  = 30,
    legendfontsize = 26,
    titlefontsize  = 10,
)
const MARKER_SIZE = 12
"""
    epoch_colors(epochs) -> Dict

Build a color map across epochs using a perceptually uniform colormap.
Used so plots auto-extend to any number of selected epochs without
hardcoded color choices.
"""
function epoch_colors(epochs)
    ints   = Int[e for e in epochs if e isa Int]
    sorted = sort(ints)
    n      = length(sorted)
    palette = cgrad(:roma, max(n, 2); categorical = true)
    return Dict(sorted[i] => palette[i] for i in 1:n)
end

"""
    plot_corr_epochs(results; lambda, kappa, error_scale=1.0,
                     noise_threshold=1.0, shift=0.1, xmax=17.0, savepath=nothing)

Plot |C(t)/C(0)| on a log scale across selected epochs. For each curve, points
are shown up to the first one whose error bar covers zero (signal lost). The
y-level of the last good point is marked with a horizontal dashed segment.
"""
function plot_corr_epochs(results;
                          lambda, kappa,
                          error_scale     = 1.0,
                          noise_threshold = 1.0,
                          shift           = 0.1,
                          xmax            = 17.0,
                          savepath        = nothing)

    p_fig = plot(;
        xlabel     = L"t",
        ylabel     = L"|C(t)/C(0)|",
        title      = L"\mathrm{Reweighted\ correlator\ across\ epochs}\ (\lambda=%$(lambda),\ \kappa=%$(kappa))",
        yscale     = :log10,
        legend     = :topright,
        framestyle = :box,
        size       = (1100, 640),
        dpi        = 200,
        left_margin = 5Plots.mm, bottom_margin = 4Plots.mm,
        PLOT_KWARGS...,
    )

    for (k, r) in enumerate(results)
        vals  = ADerrors.value.(r.corr)
        errs  = ADerrors.err.(r.corr) .* error_scale
        x     = (1:length(r.corr)) .+ (k - 1) * shift
        style = get(r, :style, :scatter)

        good = (vals .> 0) .& (vals .> noise_threshold .* errs)
        first_bad = findfirst(.!good)
        if first_bad === nothing
            last_good, add_line = length(good), false
        elseif first_bad == 1
            last_good, add_line = 0, false
        else
            last_good, add_line = first_bad - 1, true
        end

        label = r.epoch isa Integer ?
                L"\mathrm{epoch}\ %$(r.epoch)" :
                L"\mathrm{%$(r.epoch)}"

        if last_good == 0
            scatter!(p_fig, Float64[], Float64[];
                color = r.color, linecolor = r.color,
                markerstrokecolor = r.color, markersize = MARKER_SIZE,
                label = L"%$(label)\ \mathrm{(no\ signal)}")
            continue
        end

        sig_range = 1:last_good
        if style == :line
            # Theory / reference style: dashed line, no error bars
            plot!(p_fig, x[sig_range], vals[sig_range];
                color     = r.color,
                linewidth = 2.0,
                linestyle = :dash,
                marker    = :diamond,
                markersize        = MARKER_SIZE,
                markerstrokewidth = 0,
                label     = label,
            )
        else
            # Scatter with error bars
            scatter!(p_fig, x[sig_range], vals[sig_range];
                yerr              = errs[sig_range],
                color             = r.color,
                linecolor         = r.color,
                markerstrokecolor = r.color,
                label             = label,
                markersize        = MARKER_SIZE,
            )
        end

    end

    plot!(p_fig; xlims = (0.0, xmax))

    if savepath !== nothing
        savefig(p_fig, savepath * ".pdf")
    end
    return p_fig
end
"""
    plot_reweighted_correlator(models, pics, p, args; T=Float64, params_phi4=nothing)

Compute the reweighted correlator for every epoch in `args["epochs_show"]`
(filtered against `models`) and plot them on a single log-y panel.
"""
# -----------------------------------------------------------------------------
# Updated drivers: include analytic curve when λ = 0
# -----------------------------------------------------------------------------

function plot_reweighted_correlator(models, pics, p, args;
                                    T           = Float64,
                                    params_phi4 = nothing,
                                    error_scale = 1.0,
                                    xmax        = nothing,
                                    analytic    = nothing)

    isnothing(params_phi4) && error("params_phi4 required")

    epochs_show = parse_epoch_spec(args["epochs_show"])
    available   = filter(e -> haskey(models, e), epochs_show)
    isempty(available) && error("No epochs_show present in loaded models")
    if length(available) != length(epochs_show)
        @warn "Skipping missing epochs" missing=setdiff(epochs_show, available)
    end

    ncfg    = args["ncfg"]
    Neval = args["Neval"]
    tag     = ensemble_tag(p)
    colors  = epoch_colors(available)
    results = NamedTuple[]

    # Optional analytic baseline (computed by caller and passed in)
    if analytic !== nothing
        push!(results, (epoch = :analytic, corr = analytic.corr, color = "#2166ac"))
    end

    for e in available
        @info "Computing reweighted correlator" epoch=e
        res = reweighted_correlator(models[e], pics, params_phi4;
                                    N = Neval, tag = tag, T = T)
        push!(results, (epoch = e, corr = res.corr, color = colors[e]))
    end

    xmax   = something(xmax, Float64(p.L1 ÷ 2 + 1))
    saveas = joinpath(args["outdir"], "corr_epochs_" * run_stem(p) * ".png")

    fig = plot_corr_epochs(results;
        lambda          = p.lambda,
        kappa           = p.kappa,
        error_scale     = error_scale,
        noise_threshold = 1.0,
        shift           = 0.1,
        xmax            = xmax,
        savepath        = saveas)

    return fig, results
end


function plot_effective_mass_comparison(corr_results, two_point, p, args;
                                        err_scale = 1.0)

    # Separate the analytic entry (if present) from the per-epoch entries
    ana_entry = nothing
    epoch_entries = NamedTuple[]
    for r in corr_results
        if r.epoch === :analytic
            ana_entry = r
        else
            push!(epoch_entries, r)
        end
    end

    epochs = [r.epoch for r in epoch_entries]
    colors = epoch_colors(epochs)

    results = NamedTuple[]
    push!(results, (label = "two-point",     corr = two_point,        color = "#222222"))
    ana_entry !== nothing &&
        push!(results, (label = "analytic", corr = ana_entry.corr,   color = ana_entry.color))
    for r in epoch_entries
        push!(results, (label = "epoch $(r.epoch)",
                        corr  = r.corr,
                        color = colors[r.epoch]))
    end

    saveas = joinpath(args["outdir"], "meff_" * run_stem(p) * ".png")

    fig = plot_effective_mass(results;
        L_t       = p.L1,
        lambda    = p.lambda,
        kappa     = p.kappa,
        err_scale = err_scale,
        savepath  = saveas)

    # Add analytic free-theory mass reference line when λ = 0
    if p.lambda == 0.0
        m_free = sqrt(1 / p.kappa - 4)
        hline!(fig, [m_free];
            ls    = :dot,
            color = :black,
            lw    = 1.2,
            label = L"m_\mathrm{free} = %$(round(m_free, sigdigits=4))",
        )
        # Re-save with the reference line included
        base = joinpath(args["outdir"], "meff_" * run_stem(p) * ".png")
        savefig(fig, base * ".pdf")
    end

    return fig
end

# -----------------------------------------------------------------------------
# Two-point function baseline (no reweighting, translation-averaged source)
# -----------------------------------------------------------------------------

"""
    two_point_correlator(pics, p; ncfg=size(pics,4), tag=ensemble_tag(p))

Translation-averaged zero-momentum two-point correlator from the raw
configurations. Returns a `Vector{uwreal}` of length L1, normalized so that
`C(0) = 1`.

    C(t) = (1/L1) Σ_{t0} ⟨ Φ(t0) Φ(t0+t) ⟩,    Φ(t) = Σ_x φ(t,x)

Uses periodic indexing in t. The translation average reduces variance
without changing the central value.
"""
function two_point_correlator(pics, p;
                              ncfg = size(pics, 4),
                              tag  = ensemble_tag(p))
    L1 = p.L1
    # Drop singleton channel dim, take first ncfg configs
    ϕ = dropdims(pics, dims=3)[:, :, 1:ncfg]
    x = dsum(ϕ; dims = 2)                       # (L1, ncfg)
    N_cfg = size(x, 2)

    corr_per_cfg = zeros(L1, N_cfg)
    @inbounds for n in 1:N_cfg, t in 1:L1
        s = 0.0
        for t0 in 1:1
            s += x[t0, n] * x[mod1(t0 + t - 1, L1), n]
        end
        corr_per_cfg[t, n] = s / L1
    end

    corr2 = uwcorr(corr_per_cfg, tag)
    corr2 = corr2 ./ corr2[1]
    uwerr.(corr2)
    return corr2
end

# Helpers used by two_point_correlator. If they live in your `src/`, this
# include duplicates them; remove either there or here.
uwcorr(x::AbstractArray{T,2}, args...) where {T} =
    map(eachrow(x)) do r uwreal(r[:], args...) end
dsum(x; dims) = dropdims(sum(x; dims); dims)

# -----------------------------------------------------------------------------
# SNR vs epoch + two-point baseline
# -----------------------------------------------------------------------------

snr(c::uwreal) = (uwerr(c); e = ADerrors.err(c); e == 0 ? Inf : abs(ADerrors.value(c)) / e)

"""
    plot_snr_at_t(results; t, lambda, kappa, baseline=nothing, savepath=nothing)

SNR of the reweighted correlator's `t`-th point across training checkpoints.
If `baseline::uwreal` is given (typically the unreweighted two-point function
at the same `t`), its SNR is drawn as a horizontal dashed line for comparison.
"""
function plot_snr_at_t(results; t = 2,
                       lambda, kappa,
                       baseline::Union{Nothing, uwreal} = nothing,
                       savepath = nothing)

    # Only trained-epoch entries (Int epoch) go on the x-axis
    results = filter(r -> r.epoch isa Int, results)

    epochs = Int[]; snrs = Float64[]
    for r in results
        length(r.corr) < t && continue
        push!(epochs, r.epoch); push!(snrs, snr(r.corr[t]))
    end
    perm = sortperm(epochs)
    epochs, snrs = epochs[perm], snrs[perm]

    p_fig = plot(epochs, snrs;
        xlabel     = "Epoch",
        ylabel     = L"|C(t{=}%$(t-1))| / \sigma_{C(t{=}%$(t-1))}",
        title      = L"\mathrm{SNR\ at}\ t = %$(t-1)\ (\lambda=%$(lambda),\ \kappa=%$(kappa))",
        marker     = :circle,
        markersize = MARKER_SIZE,
        markerstrokewidth = 0,
        lw         = 1.8,
        color      = :steelblue,
        label      = "trained flow",
        yscale     = :log10,
        framestyle = :box,
        size       = (1100, 580),
        dpi        = 200,
        left_margin = 5Plots.mm, bottom_margin = 4Plots.mm,
        PLOT_KWARGS...,
    )

    if baseline !== nothing
        snr_2pt = snr(baseline)
        hline!(p_fig, [snr_2pt];
            ls    = :dash,
            color = :black,
            lw    = 1.5,
            label = L"\mathrm{two-point\ baseline}\ (\mathrm{SNR}=%$(round(snr_2pt, sigdigits=3)))",
        )
    end

    # Optional SNR=1 reference; only useful if SNRs span the unit threshold
    if minimum(snrs) < 5 && (baseline === nothing || snr(baseline) < 5)
        hline!(p_fig, [1.0]; ls = :dot, color = :gray, lw = 1.0, label = "")
        annotate!(p_fig, epochs[1], 1.1,
            text("SNR = 1", 8, :left, :bottom, :gray))
    end

    if savepath !== nothing
        savefig(p_fig, savepath * ".pdf")
    end
    return p_fig
end

"""
    plot_snr_vs_epoch(corr_results, two_point, p, args; t=args["snr_t"])

Driver that wires the per-epoch reweighted-correlator `results` together with
the two-point baseline and writes to <outdir>/snr_t<t>_<run_stem>.
"""
function plot_snr_vs_epoch(corr_results, two_point, p, args;
                           t = nothing)
    t = something(t, args["snr_t"])
    baseline = length(two_point) >= t ? two_point[t] : nothing

    saveas = joinpath(args["outdir"], "snr_t$(t-1)_" * run_stem(p) * ".png")

    return plot_snr_at_t(corr_results;
        t        = t,
        lambda   = p.lambda,
        kappa    = p.kappa,
        baseline = baseline,
        savepath = saveas)
end
# -----------------------------------------------------------------------------
# Effective mass (cosh definition on a periodic lattice)
# -----------------------------------------------------------------------------
##
"""
effective_mass_cosh(corr::Vector{uwreal}, L_t; tmax=L_t÷2)

Cosh effective mass: solve  C(t+1)/C(t) = cosh(m(t+1-L_t/2)) / cosh(m(t-L_t/2))
at each t. Uses `Roots.find_zero` for the central value and the implicit
function theorem (analytic dm/dratio) to propagate the `uwreal`.
effective_mass_acosh(corr::Vector{uwreal}, L_t; tmax=L_t÷2-1)

Effective mass on a periodic lattice using the three-point formula:
    m_eff(t) = acosh( (C(t+1) + C(t-1)) / (2 C(t)) )

Valid for 1 < t < L_t - 1, gives a constant plateau at the true mass for
a single-cosh correlator (away from boundaries). Errors propagate through
the `uwreal` arithmetic automatically.

Returns a Vector{uwreal} indexed 1..tmax, where entry t corresponds to
m_eff at lattice time t (1-based; entry 1 = m_eff at lattice t=2).
"""
function effective_mass_acosh(corr::Vector{uwreal}, L_t::Int; tmax = L_t ÷ 2 - 1)
    tmax = min(tmax, length(corr) - 2)            # need corr[t-1], corr[t], corr[t+1]
    meff = Vector{uwreal}(undef, tmax)

    for t in 1:tmax
        # Map output index t to lattice index t+1 so we use corr[t], corr[t+1], corr[t+2]
        c_minus = corr[t]
        c_t     = corr[t + 1]
        c_plus  = corr[t + 2]

        ratio   = (c_plus + c_minus) / (2 * c_t)
        rv      = ADerrors.value(ratio)

        if !(rv ≥ 1.0) || !isfinite(rv)
            meff[t] = uwreal(NaN)
            continue
        end

        # acosh(ratio) directly — uwreal arithmetic handles error propagation
        # via the chain rule. acosh isn't always overloaded for uwreal, so use
        # the identity acosh(x) = log(x + sqrt(x^2 - 1)).
        meff[t] = log(ratio + sqrt(ratio * ratio - 1))
    end
    return meff
end


function effective_mass_cosh(corr::Vector{uwreal}, L_t::Int; tmax = L_t ÷ 2)
    tmax = min(tmax, length(corr) - 1)
    meff = Vector{uwreal}(undef, tmax)

    for t in 1:tmax
        ratio = corr[t+1] / corr[t]
        rv    = ADerrors.value(ratio)
        a, b  = t - L_t / 2, (t + 1) - L_t / 2
        println(rv)
        if !(0 < rv ≤ 1.0)
            meff[t] = uwreal(NaN); continue
        end

        m_central = try
            find_zero(m -> cosh(m * b) / cosh(m * a) - rv,
                      (1e-6, 5.0), Roots.Bisection())
        catch e
            println("caugth:",e)
            NaN
        end

        if isnan(m_central)
            meff[t] = uwreal(NaN)
        else
            num, den   = cosh(m_central * b), cosh(m_central * a)
            dnum, dden = b * sinh(m_central * b), a * sinh(m_central * a)
            dr_dm      = (dnum * den - num * dden) / den^2
            meff[t]    = m_central + (ratio - rv) / dr_dm
        end
    end
    return meff
end

##

"""
    plot_effective_mass(results; L_t, lambda, kappa, tmax=nothing,
                        noise_threshold=1.0, xmax=nothing, err_scale=1.0,
                        savepath=nothing)

Plot cosh effective mass for each entry in `results`, dropping points once
m_eff loses signal (NaN, non-positive, or SNR below threshold). Cutoff is
marked with a horizontal dashed segment in the curve's color.
"""
function plot_effective_mass(results;
                             L_t, lambda, kappa,
                             tmax            = nothing,
                             noise_threshold = 1.0,
                             xmax            = nothing,
                             err_scale       = 1.0,
                             savepath        = nothing)

    tmax = something(tmax, L_t ÷ 2 - 1)
    xmax = something(xmax, Float64(tmax) + 1.5)

    p_fig = plot(;
        xlabel     = L"t",
        ylabel     = L"m_\mathrm{eff}(t)",
        title      = L"\mathrm{Effective\ mass}\ (\lambda=%$(lambda),\ \kappa=%$(kappa))",
        legend     = :topright,
        framestyle = :box,
        size       = (1100, 640),
        dpi        = 200,
        left_margin = 5Plots.mm, bottom_margin = 4Plots.mm,
        PLOT_KWARGS...,
    )

    for (k, r) in enumerate(results)
        meff = effective_mass_acosh(r.corr, L_t; tmax = tmax)
        uwerr.(meff)
        vals  = ADerrors.value.(meff)
        errs  = ADerrors.err.(meff) .* err_scale
        # Lattice time corresponding to each meff entry is t+1 (centered):
        x     = (2:(tmax+1)) .+ (k - 1) * 0.1
        style = get(r, :style, :scatter)

        good = isfinite.(vals) .& (vals .> 0) .& (vals .> noise_threshold .* errs)
        first_bad = findfirst(.!good)
        last_good = first_bad === nothing ? length(good) :
                    first_bad == 1       ? 0             :
                                           first_bad - 1

        if last_good == 0
            scatter!(p_fig, Float64[], Float64[];
                color = r.color, linecolor = r.color,
                markerstrokecolor = r.color, markersize = MARKER_SIZE,
                label = "$(r.label) (unresolved)")
            continue
        end

        sig_range = 1:last_good
        if style == :line
            plot!(p_fig, x[sig_range], vals[sig_range];
                color     = r.color,
                linewidth = 1.0,
                linestyle = :dash,
                marker    = :diamond,
                markersize        = 0,
                markerstrokewidth = 0,
                label     = "$(r.label)",
            )
        else
            scatter!(p_fig, x[sig_range], vals[sig_range];
                yerr              = errs[sig_range],
                color             = r.color,
                linecolor         = r.color,
                markerstrokecolor = r.color,
                markersize        = MARKER_SIZE,
                label             = "$(r.label)",
            )
        end
    end

    plot!(p_fig; xlims = (1.5, xmax))

    if savepath !== nothing
        savefig(p_fig, savepath * ".pdf")
    end
    return p_fig
end
"""
    plot_effective_mass_comparison(corr_results, two_point, p, args;
                                   err_scale=1.0)

Driver: builds effective-mass curves for the two-point baseline and each
epoch in `corr_results`, then plots them on a shared axis.
"""
function plot_effective_mass_comparison(corr_results, two_point, p, args;
                                        analytic  = nothing,
                                        err_scale = 1.0)

    trained = filter(r -> r.epoch isa Int, corr_results)
    isempty(trained) && error("No trained-epoch entries in corr_results")
    last_epoch = maximum(r.epoch for r in trained)
    last_entry = first(filter(r -> r.epoch == last_epoch, trained))

    results = NamedTuple[
        (label = "two-point", corr = two_point, color = "#222222"),
    ]
    if analytic !== nothing
        push!(results, (label = "analytic",
                        corr  = analytic.corr,
                        color = "#2166ac",
                        style = :line))          
    end
    push!(results, (label = "epoch $(last_epoch)",
                    corr  = last_entry.corr,
                    color = "#d6604d"))

    saveas = joinpath(args["outdir"], "meff_" * run_stem(p))

    return plot_effective_mass(results;
        L_t       = p.L1,
        lambda    = p.lambda,
        kappa     = p.kappa,
        err_scale = err_scale,
        savepath  = saveas)
end

# -----------------------------------------------------------------------------
# Analytic free-theory reweighting (λ = 0 only)
# -----------------------------------------------------------------------------

"""
    reweighted_correlator_ana(p, pics, params_phi4; N=10000, T=Float64,
                              tag="reweighting_ana_<λ>_<κ>")

Reweighted correlator using the *analytic* free-theory flow:
    f(p) = 1 / (p̂² + m²),  m² = 1/κ - 4
projected onto the source slice. Only meaningful for λ = 0; the function
ignores `params_phi4.lambda` for the analytic part but still uses the full
φ⁴ action for the Boltzmann weight, so the reweighting identity stays valid.

Returns `(corr, WT)` matching `reweighted_correlator`.
"""
function reweighted_correlator_ana(p, pics, params_phi4;
                                   N   = 10000,
                                   T   = Float64,
                                   tag = "reweighting_ana_$(p.lambda)_$(p.kappa)")

    L1, L2 = p.L1, p.L2
    @assert size(pics, 4) ≥ N "Need at least $N configurations, got $(size(pics, 4))"

    shuffled = shuffle_data(pics)[1]
    ϕ_test   = shuffled[:, :, :, 1:N]

    # Analytic free-theory propagator on the lattice
    m²  = 1 / p.kappa - 4
    fp  = [inv(T(1) * ((T(2) * sin(p_mom / T(2)))^2 + m²))
           for p_mom in (T(2π) / T(L1)) .* T.(0:(L1 - 1))]
    Fp  = zeros(T, L1, L2);   Fp[:, 1] = fp
    Fn  = ifft(Fp) * L2 / (2 * p.kappa)
    fn  = reshape(real.(Fn), L1, L2, 1)

    # Build the Series-valued field and its flowed copy
    PHI  = Array{Series{Float64, 2}, 3}(undef, L1, L2, N)
    PHIT = Array{Series{Float64, 2}, 3}(undef, L1, L2, N)
    @info "Constructing PHI/PHIT for analytic reweighting" N=N
    for n in 1:N, i in 1:L1, j in 1:L2
        PHI[i, j, n]  = Series{Float64, 2}((ϕ_test[i, j, 1, n], zero(T)))
        PHIT[i, j, n] = Series{Float64, 2}((ϕ_test[i, j, 1, n], fn[i, j, 1]))
    end

    PHIT_T      = sum(PHIT, dims=2)[:, 1, :]
    action_term = (ActionPhi4(PHI,  params_phi4.kappa, params_phi4.lambda)[:] .-
                   ActionPhi4(PHIT, params_phi4.kappa, params_phi4.lambda)[:])
    source_term = sum(PHIT[1, :, :], dims=1)[1, :]

    ϵ   = Series{T, 2}((zero(Float64), one(Float64)))
    WT  = exp.(action_term .+ ϵ .* source_term)

    mean_phi_t_trw = Vector{Series{uwreal, 2}}(undef, L1)
    for t in 1:L1
        mean_phi_t_trw[t] = uwreal(WT .* PHIT_T[t, :], tag) / uwreal(WT, tag)
    end

    corr_trw  = [mean_phi_t_trw[t].c[2] for t in 1:L1]
    corr_norm = corr_trw ./ corr_trw[1]
    uwerr.(corr_norm)

    return (corr = corr_norm, WT = WT)
end

"""
    plot_final_correlator(corr_results, two_point, p, args;
                         analytic=nothing, error_scale=1.0, xmax=nothing)

Headline correlator figure: two-point baseline, analytic free-theory reweighting
(when λ=0), and the trained flow at the last epoch in `corr_results`. Same
visual conventions as the per-epoch comparison plot.
"""
function plot_final_correlator(corr_results, two_point, p, args;
                               analytic    = nothing,
                               error_scale = 1.0,
                               xmax        = nothing)

    trained = filter(r -> r.epoch isa Int, corr_results)
    isempty(trained) && error("No trained-epoch entries in corr_results")
    last_epoch = maximum(r.epoch for r in trained)
    last_entry = first(filter(r -> r.epoch == last_epoch, trained))

    results = NamedTuple[
        (epoch = "two-point", corr = two_point, color = "#222222"),
    ]
    if analytic !== nothing
        push!(results, (epoch = "analytic",
                        corr  = analytic.corr,
                        color = "#2166ac",
                        style = :line))               # ← dashed line, no error bars
    end
    push!(results, (epoch = "epoch $(last_epoch)",
                    corr  = last_entry.corr,
                    color = "#d6604d"))

    xmax   = something(xmax, Float64(p.L1 ÷ 2 + 1))
    saveas = joinpath(args["outdir"], "corr_final_" * run_stem(p))

    return plot_corr_epochs(results;
        lambda          = p.lambda,
        kappa           = p.kappa,
        error_scale     = error_scale,
        noise_threshold = 1.0,
        shift           = 0.1,
        xmax            = xmax,
        savepath        = saveas)
end
"""
    correlator_variance(corr::Vector{uwreal}) -> Vector{Float64}

Per-timeslice variance σ²(t) of a `uwreal` correlator. Just `err(t)^2`
after ensuring `uwerr` has populated the cache.
"""
function correlator_variance(corr::Vector{uwreal})
    uwerr.(corr)
    return ADerrors.err.(corr) .^ 2
end

"""
    plot_variances(two_point, trained_corr, p, args; analytic=nothing,
                   xmax=nothing, savepath=nothing)

Overlay σ²(t) of the two-point baseline, the trained correlator at the last
epoch, and (optionally) the analytic free-theory reweighted correlator.
Y-axis is log10.
"""
function plot_variances(two_point, trained_corr, p, args;
                        analytic = nothing,
                        xmax     = nothing,
                        savepath = nothing)

    L1 = p.L1
    xmax = something(xmax, Float64(L1 ÷ 2 + 1))

    ts = 2:L1
    var_tp = correlator_variance(two_point)[ts]
    var_tr = correlator_variance(trained_corr)[ts]

    # Collect all variances to determine y-range and tick placement
    all_vars = vcat(var_tp, var_tr)
    if analytic !== nothing
        var_an = correlator_variance(analytic.corr)[ts]
        append!(all_vars, var_an)
    end
    pos_vars = filter(x -> isfinite(x) && x > 0, all_vars)

    lo_exp = floor(Int, log10(minimum(pos_vars)))
    hi_exp = ceil(Int,  log10(maximum(pos_vars)))
    # One tick every 2 decades if range is large
    step = (hi_exp - lo_exp) > 8 ? 2 : 1
    tick_exps  = lo_exp:step:hi_exp
    tick_vals  = [10.0^e for e in tick_exps]
    tick_labs  = [L"10^{%$e}" for e in tick_exps]

    p_fig = plot(;
        xlabel     = L"t",
        ylabel     = L"\sigma^2(C(t))",
        title      = L"\mathrm{Per\!-\!timeslice\ variance}\ (\lambda=%$(p.lambda),\ \kappa=%$(p.kappa))",
        yscale     = :log10,
        ylims      = (10.0^lo_exp / 3, 10.0^hi_exp * 3),
        yticks     = (tick_vals, tick_labs),
        minorgrid  = false,                          # disable to avoid the NaN crash
        legend     = :left,
        framestyle = :box,
        size       = (1100, 620), dpi = 200,
        left_margin = 5Plots.mm, bottom_margin = 4Plots.mm,
        PLOT_KWARGS...,
    )

    plot!(p_fig, collect(ts), var_tp;
        lw = 2.0, color = "#222222",
        marker = :circle, markersize = MARKER_SIZE, markerstrokewidth = 0,
        label = "two-point")

    if analytic !== nothing
        plot!(p_fig, collect(ts), correlator_variance(analytic.corr)[ts];
            lw = 2.0, color = "#2166ac",
            marker = :diamond, markersize = MARKER_SIZE, markerstrokewidth = 0,
            label = "analytic")
    end

    plot!(p_fig, collect(ts), var_tr;
        lw = 2.0, color = "#d6604d",
        marker = :square, markersize = MARKER_SIZE, markerstrokewidth = 0,
        label = "trained (last epoch)")

    plot!(p_fig; xlims = (1.5, xmax + 0.5))

    if savepath !== nothing
        savefig(p_fig, savepath * ".pdf")
    end
    return p_fig
end

"""
    plot_variance_ratio(two_point, trained_corr, p, args; analytic=nothing,
                        xmax=nothing, savepath=nothing)

Ratio σ²_trained(t) / σ²_two-point(t). Values < 1 indicate variance reduction
from the trained flow. Includes analytic free-theory ratio when supplied.
"""
function plot_variance_ratio(two_point, trained_corr, p, args;
                             analytic = nothing,
                             xmax     = nothing,
                             savepath = nothing)

    L1   = p.L1
    xmax = something(xmax, Float64(L1 ÷ 2 + 1))
    ts   = 2:L1
    ts_v = collect(ts)

    var_tp = correlator_variance(two_point)[ts]
    var_tr = correlator_variance(trained_corr)[ts]
    ratio_tr = var_tr ./ max.(var_tp, eps())

    # All ticks/ranges computed from positive finite values
    all_ratios = copy(ratio_tr)
    if analytic !== nothing
        var_an   = correlator_variance(analytic.corr)[ts]
        ratio_an = var_an ./ max.(var_tp, eps())
        append!(all_ratios, ratio_an)
    end
    pos = filter(x -> isfinite(x) && x > 0, all_ratios)

    lo_exp = floor(Int, log10(minimum(pos)))
    hi_exp = ceil(Int,  log10(maximum(pos)))
    step   = (hi_exp - lo_exp) > 8 ? 2 : 1
    tick_exps = lo_exp:step:hi_exp
    tick_vals = [10.0^e for e in tick_exps]
    tick_labs = [L"10^{%$e}" for e in tick_exps]

    p_fig = plot(;
        xlabel     = L"t",
        ylabel     = L"\sigma^2_\mathrm{flow}(t)\ /\ \sigma^2_\mathrm{two-point}(t)",
        title      = L"\mathrm{Variance\ reduction}\ (\lambda=%$(p.lambda),\ \kappa=%$(p.kappa))",
        yscale     = :log10,
        ylims      = (10.0^lo_exp / 3, 10.0^hi_exp * 3),
        yticks     = (tick_vals, tick_labs),
        minorgrid  = false,
        legend     = :bottomleft,
        framestyle = :box,
        size       = (1100, 620), dpi = 200,
        left_margin = 5Plots.mm, bottom_margin = 4Plots.mm,
        PLOT_KWARGS...,
    )
    plot!(p_fig; minorgrid = false)

    if analytic !== nothing
        var_an   = correlator_variance(analytic.corr)[ts]
        ratio_an = var_an ./ max.(var_tp, eps())
        plot!(p_fig, ts_v, ratio_an;
            lw = 2.0, color = "#2166ac",
            marker = :diamond, markersize = MARKER_SIZE, markerstrokewidth = 0,
            label = "analytic / two-point")
    end

    plot!(p_fig, ts_v, ratio_tr;
        lw = 2.0, color = "#d6604d",
        marker = :square, markersize = MARKER_SIZE, markerstrokewidth = 0,
        label = "trained / two-point")

    hline!(p_fig, [1.0]; ls = :dash, color = :gray, lw = 1.2, label = "")

    plot!(p_fig; xlims = (1.5, xmax + 0.5))

    if savepath !== nothing
        savefig(p_fig, savepath * ".pdf")
    end
    return p_fig
end

"""
    plot_variance_diagnostics(corr_results, two_point, p, args; analytic=nothing)

Driver: writes both variance plots using the trained correlator at the last
epoch and the two-point baseline.
"""
function plot_variance_diagnostics(corr_results, two_point, p, args;
                                   analytic = nothing)
    trained = filter(r -> r.epoch isa Int, corr_results)
    isempty(trained) && error("No trained-epoch entries")
    last_epoch = maximum(r.epoch for r in trained)
    last_entry = first(filter(r -> r.epoch == last_epoch, trained))

    stem = run_stem(p)

    plot_variances(two_point, last_entry.corr, p, args;
        analytic = analytic,
        savepath = joinpath(args["outdir"], "variance_" * stem) )

    plot_variance_ratio(two_point, last_entry.corr, p, args;
        analytic = analytic,
        savepath = joinpath(args["outdir"], "variance_ratio_" * stem)  )

    return nothing
end

"""
    save_correlator_data(corr_results, two_point, p, args; analytic=nothing)

Save the raw correlator data to <outdir>/correlators_<run_stem>.jld2 so plots
can be regenerated without re-running the reweighting computation.

Layout inside the .jld2:
  - two_point      :: Vector{uwreal}
  - corr_results   :: Vector{NamedTuple}  with fields (epoch, corr, color)
  - analytic       :: NamedTuple (corr, WT) or nothing
  - p              :: NamedTuple of run hyperparameters
  - timestamp      :: String (ISO-8601)
"""
function save_correlator_data(corr_results, two_point, p, args;
                              analytic = nothing)

    fname = joinpath(args["outdir"], "correlators_" * run_stem(p) * ".jld2")

    # Strip color from corr_results — it's a display concern, not data.
    # Keep epoch + corr as the canonical record.
    corr_data = [(epoch = r.epoch, corr = r.corr) for r in corr_results]

    JLD2.jldopen(fname, "w") do file
        file["two_point"]    = two_point
        file["corr_results"] = corr_data
        file["analytic"]     = analytic === nothing ? nothing : analytic.corr
        file["p"]            = p
        file["timestamp"]    = string(Dates.now())
    end

    @info "Saved correlator data" file=fname size_kb=round(filesize(fname)/1024, digits=1)
    return fname
end

# -----------------------------------------------------------------------------
# Main — for now just verify loading works end-to-end
# -----------------------------------------------------------------------------
##
function main()
    args = parse_commandline()
    mkpath(args["outdir"])

    # Bundle hyperparameters into a NamedTuple for ergonomic threading
    p = (
        lambda = args["lambda"], kappa = args["kappa"],
        L1     = args["L1"],     L2    = args["L2"],
        depth  = args["depth"],  nodes = args["nodes"],
        bs     = args["bs"],     activ = args["activ"],
        lr     = args["lr"],
    )
    phi4_params = Phi4_params(p.kappa, p.lambda)

    @info "Run configuration" p

    model = make_model1((32, 8), 0.2485; nodes=4, activ=relu)



    epochs_to_load = parse_epoch_spec(args["epochs_load"])
    epochs_to_show = parse_epoch_spec(args["epochs_show"])

    run    = load_run(p, args)
    models = load_models(p, args, epochs_to_load, model)
    pics   = load_configurations(args, p)

    pics = pics[:,:,:,args["thermalize"]:end]
    pics = pics[:,:,:,1:args["Neval"]]
    source = uwreal(sum(pics[1,:,:,:],dims=(1,2))[1,1,:],ensemble_tag(p)); uwerr(source)
    @info "Loaded successfully" begin
        n_epochs_in_models = length(models)
        first_few_epochs   = sort(collect(keys(models)))[1:min(end, 8)]
        n_cfg              = size(pics, 4)
    end

    # Verify the epochs_to_show subset actually exists in models
    missing_show = setdiff(epochs_to_show, keys(models))
    isempty(missing_show) || @warn "Some --epochs-show not present in loaded models" missing_show

        # ------------------------------------------------------------------
    # Plot 1: model output mean / variance across epochs
    # ------------------------------------------------------------------
    if true
        plot_field_stats(models, pics, p, args; ncfg=size(pics,4))
    end


    # Analytic baseline if λ = 0
    analytic = nothing
    if p.lambda == 0.0
        @info "Computing analytic free-theory reweighting baseline"
        analytic = reweighted_correlator_ana(p, pics, phi4_params ;
                                             N = args["Neval"],
                                             T = Float64)
    else
        @info "λ ≠ 0, skipping analytic baseline" lambda=p.lambda
    end

    # ------------------------------------------------------------------
    # Plot 2: reweighted correlator at selected epochs (+ analytic if λ=0)
    # ------------------------------------------------------------------
    _, corr_results = plot_reweighted_correlator(models, pics, p, args;
        T           = Float64,
        params_phi4 = phi4_params ,
        error_scale = 1.0,
        #analytic    = analytic,
    )

    # ------------------------------------------------------------------
    # Plot 3: two-point baseline
    # ------------------------------------------------------------------
    @info "Computing two-point baseline"
    two_point = two_point_correlator(pics, p; ncfg = args["Neval"])

    # ------------------------------------------------------------------
    # Plot 4: SNR vs epoch (with two-point baseline reference)
    # ------------------------------------------------------------------
    plot_snr_vs_epoch(corr_results, two_point, p, args)
    ##
    # ------------------------------------------------------------------
    # Plot 5: effective mass comparison
    # ------------------------------------------------------------------
    plot_effective_mass_comparison(corr_results, two_point, p, args; 
					analytic = analytic)
    ##
    # ------------------------------------------------------------------
    # Plot 6: final correlator (two-point + analytic if λ=0 + last epoch)
    # ------------------------------------------------------------------
    plot_final_correlator(corr_results, two_point, p, args;
                            analytic = analytic)
    ##
    # ------------------------------------------------------------------
    # Plot 6: variance and variance-reduction diagnostics
    # ------------------------------------------------------------------
    plot_variance_diagnostics(corr_results, two_point, p, args;
                                )
    save_correlator_data(corr_results, two_point, p, args; analytic = analytic)
    return (; args, p, run, models, pics, epochs_to_load, epochs_to_show)


end

isinteractive() || main()
 file or directory

