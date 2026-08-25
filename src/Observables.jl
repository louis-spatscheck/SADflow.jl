# src/observables.jl
#
# Physics observables extracted from raw configs or from a trained flow:
# two-point correlators (direct + FFT), effective mass, and importance-
# reweighted correlators. This is the "analysis" half of the pipeline —
# scripts/generate_plots.jl (from plot_IT3.jl) should import these rather
# than redefining them.
#
# Needs `using Roots` for `effective_mass_cosh`'s find_zero call — add
# Roots to Project.toml if it isn't already a dependency.


dsum(x; dims) = dropdims(sum(x; dims=dims); dims=dims)

uwcorr(x::AbstractArray{T,2}, args...) where {T} =
    map(eachrow(x)) do r
        ADerrors.uwreal(r[:], args...)
    end
"""
    correlator(interpol_chain, tag, niter_vec; wpm=Dict(), ed=100, timeav=false)

`uwreal`-valued autocorrelation of `interpol_chain` (shape `(L, chainsize)`)
around its own mean, either at each Monte Carlo "time slice" directly
(`timeav=false`) or time-averaged over all cyclic shifts (`timeav=true`).
Source: training_5.jl:412-451.
"""
function correlator(interpol_chain::Array{T,2}, tag::String, niter_vec::Vector{Int64};
                     wpm=Dict{String,Vector{Float64}}(), ed=100, timeav=false) where T
    L = size(interpol_chain, 1)
    chainsize = size(interpol_chain, 2)

    mean_uw = Vector{ADerrors.uwreal}(undef, L)
    for i in 1:L
        mean_uw[i] = ADerrors.uwreal(interpol_chain[i, :], tag, niter_vec)
    end

    dif_chain = similar(interpol_chain)
    for k in 1:chainsize
        dif_chain[:, k] .= interpol_chain[:, k] .- ADerrors.value.(mean_uw)
    end

    ct = Vector{ADerrors.uwreal}(undef, L)
    if !timeav
        for i in 1:L
            ct[i] = ADerrors.uwreal(dif_chain[i, :] .* dif_chain[1, :], tag, niter_vec)
        end
    else
        for t in 1:L
            ct[t] = ADerrors.uwreal(0.0)
            for ti in 1:L
                tt = ti + t
                tt = tt > L ? tt - L : tt
                ct[t] += ADerrors.uwreal(dif_chain[tt, :] .* dif_chain[ti, :] ./ L, tag, niter_vec)
            end
        end
        ct = reverse(ct)
    end
    return ct
end

"""
    two_point_corr_fft(pics)

Zero-momentum two-point correlator via FFT convolution theorem, averaged
over configs. `pics` is `(Lx, Ly, 1, N)`.
"""
function two_point_corr_fft(pics)
    Lx, Ly, _, N = size(pics)
    C = similar(pics, eltype(pics), Lx, Ly)
    fill!(C, zero(eltype(pics)))
    for n in 1:N
        φ = copy(pics[:, :, 1, n])
        φ .-= mean(φ)
        F = fft(φ)
        C .+= real.(ifft(abs2.(F))) / (Lx * Ly)
    end
    C ./= N
    return C
end

"""
    two_point_correlator(pics, L1; ncfg=size(pics,4), tag="ensemble")

Translation-averaged zero-momentum two-point correlator with `uwreal`
error propagation, normalized so `C(0) = 1`:

    C(t) = (1/L1) Σ_{t0} ⟨ Φ(t0) Φ(t0+t) ⟩,    Φ(t) = Σ_x φ(t,x)

Source: plot_IT3.jl:1035-1057. NOTE: the original signature took a `p`
"args" struct for `L1`/`tag` (`ensemble_tag(p)`) that isn't part of the
material reviewed so far — signature below takes `L1`/`tag` directly;
adapt call sites in scripts/generate_plots.jl accordingly, or reintroduce
whatever config struct `plot_IT3.jl` used once you've pulled the rest of
that file in.
"""
function two_point_correlator(pics, L1::Int; ncfg=size(pics, 4), tag="ensemble")
    ϕ = dropdims(pics, dims=3)[:, :, 1:ncfg]
    x = dsum(ϕ; dims=2)                       # (L1, ncfg)
    N_cfg = size(x, 2)

    corr_per_cfg = zeros(L1, N_cfg)
    @inbounds for n in 1:N_cfg, t in 1:L1
        corr_per_cfg[t, n] = x[1, n] * x[mod1(t, L1), n] / L1
    end

    corr2 = uwcorr(corr_per_cfg, tag)
    corr2 = corr2 ./ corr2[1]
    ADerrors.uwerr.(corr2)
    return corr2
end

"""
    reweighted_correlator(model, prior, params; N=2000, tag="rw", trace=true, ns_trace=25, rng=Random.default_rng())

Importance-reweighted correlator using the trained flow's implicit
Jacobian trace (via `FormalSeries` dual-number bookkeeping), returning
`(corr=normalized_correlator, WT=reweighting_factors)`. Set `trace=false`
to skip the (expensive) trace term for a cheaper approximate estimate.
Source: plot_IT3.jl:739-785.
"""
function reweighted_correlator(model, prior, params::Phi4Params; N=2000, tag="rw", T=Float64, trace=true,
                               ns_trace=25, rng=Random.default_rng())
    L1, L2 = size(prior, 1), size(prior, 2)
    @assert size(prior, 4) ≥ N "Need at least $N configurations, got $(size(prior, 4))"

    ϕ_test  = prior[:, :, :, 1:N]
    f_model = model(ϕ_test)

    PHI  = Array{FormalSeries.Series{Float64,2},3}(undef, L1, L2, N)
    PHIT = Array{FormalSeries.Series{Float64,2},3}(undef, L1, L2, N)
    for n in 1:N, i in 1:L1, j in 1:L2
        PHI[i, j, n]  = FormalSeries.Series{Float64,2}((ϕ_test[i, j, 1, n], zero(T)))
        PHIT[i, j, n] = FormalSeries.Series{Float64,2}((ϕ_test[i, j, 1, n], f_model[i, j, 1, n]))
    end

    PHIT_T = sum(PHIT, dims=2)[:, 1, :]

    ϵ = FormalSeries.Series{T,2}((zero(Float64), one(Float64)))
    action_term = (action(PHI, params)[:] .-
                   action(PHIT, params)[:])
    source_term = sum(PHIT[1, :, :], dims=1)[1, :]

    if trace
        trace_term = mean(trJ(rng, model, ϕ_test; ns=ns_trace), dims=2)[:]
        WT = exp.(action_term .+ ϵ .* source_term .+ ϵ .* trace_term)
    else
        WT = exp.(action_term .+ ϵ .* source_term)
    end

    mean_phi_t_trw = Vector{FormalSeries.Series{ADerrors.uwreal,2}}(undef, L1)
    for t in 1:L1
        mean_phi_t_trw[t] = ADerrors.uwreal(WT .* PHIT_T[t, :], tag) / ADerrors.uwreal(WT, tag)
    end

    corr_trw  = [mean_phi_t_trw[t].c[2] for t in 1:L1]
    corr_norm = corr_trw ./ corr_trw[1]
    ADerrors.uwerr.(corr_norm)

    return (corr=corr_norm, WT=WT)
end

"""
    effective_mass_acosh(corr, L_t; tmax=L_t÷2-1)

Effective mass via the three-point formula
`m_eff(t) = acosh((C(t+1)+C(t-1)) / (2 C(t)))`, valid for `1 < t < L_t-1`.
Errors propagate through `uwreal` arithmetic automatically (implemented as
`log(ratio + sqrt(ratio²-1))` since `acosh` isn't overloaded for `uwreal`).
Source: plot_IT3.jl:1177-1201.
"""
function effective_mass_acosh(corr::Vector{ADerrors.uwreal}, L_t::Int; tmax=L_t ÷ 2 - 1)
    tmax = min(tmax, length(corr) - 2)
    meff = Vector{ADerrors.uwreal}(undef, tmax)

    for t in 1:tmax
        c_minus = corr[t]
        c_t     = corr[t + 1]
        c_plus  = corr[t + 2]

        ratio = (c_plus + c_minus) / (2 * c_t)
        rv    = ADerrors.value(ratio)

        if !(rv ≥ 1.0) || !isfinite(rv)
            meff[t] = ADerrors.uwreal(NaN)
            continue
        end
        meff[t] = log(ratio + sqrt(ratio * ratio - 1))
    end
    return meff
end

"""
    effective_mass_cosh(corr, L_t; tmax=L_t÷2)

Cosh effective mass: solves `C(t+1)/C(t) = cosh(m·b)/cosh(m·a)` for `m` at
each `t` (via `Roots.find_zero`, needs `using Roots`), then propagates the
`uwreal` error through the implicit function theorem (analytic `dm/dratio`).
Source: plot_IT3.jl:1204-1235.
"""
function effective_mass_cosh(corr::Vector{ADerrors.uwreal}, L_t::Int; tmax=L_t ÷ 2)
    tmax = min(tmax, length(corr) - 1)
    meff = Vector{ADerrors.uwreal}(undef, tmax)

    for t in 1:tmax
        ratio = corr[t + 1] / corr[t]
        rv = ADerrors.value(ratio)
        a, b = t - L_t / 2, (t + 1) - L_t / 2

        if !(0 < rv ≤ 1.0)
            meff[t] = ADerrors.uwreal(NaN)
            continue
        end

        m_central = try
            Roots.find_zero(m -> cosh(m * b) / cosh(m * a) - rv, (1e-6, 5.0), Roots.Bisection())
        catch
            NaN
        end

        if isnan(m_central)
            meff[t] = ADerrors.uwreal(NaN)
        else
            num, den   = cosh(m_central * b), cosh(m_central * a)
            dnum, dden = b * sinh(m_central * b), a * sinh(m_central * a)
            dr_dm      = (dnum * den - num * dden) / den^2
            meff[t]    = m_central + (ratio - rv) / dr_dm
        end
    end
    return meff
end

"""
    correlator_variance(corr)

Per-time-slice variance `σ²(t) = err(C(t))²` of a `uwreal` correlator, for
SNR/variance-comparison plots. Source: plot_IT3.jl:1476-1479.
"""
function correlator_variance(corr::Vector{ADerrors.uwreal})
    ADerrors.uwerr.(corr)
    return ADerrors.err.(corr) .^ 2
end

"""
    estimate_source_variance(phi)

Estimate the variance of the time-slice source used by the KL loss.
"""
function estimate_source_variance(phi)
    x = vec(sum(phi[1, :, :, :], dims=1))
    μ = mean(x)

    return mean((x .- μ) .^ 2)
end