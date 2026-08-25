# src/autodiff.jl


"""
    fft(x::AbstractArray{<:Dual}, dims)
    ifft(x::AbstractArray{<:Dual}, dims)

FFT/IFFT of a `ForwardDiff.Dual`-valued array
"""
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

# --- ADerrors <-> FormalSeries glue ------------------------------------

import Base: broadcastable
Base.broadcastable(s::ADerrors.uwreal) = Ref(s)

function ADerrors.uwreal(obs::Vector{FormalSeries.Series{T,N}},
                          ID::String,
                          rep=nothing) where {T,N}
    if rep === nothing
        rep = [length(obs)]
    end
    return FormalSeries.Series{ADerrors.uwreal,N}(
        ntuple(i -> ADerrors.uwreal([real(obs[j].c[i]) for j in 1:length(obs)], ID, rep), N)
    )
end

function ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal,N},
                         wpm::Union{Dict{Int64,Vector{Float64}},Dict{String,Vector{Float64}}}) where N
    for i in 1:N
        ADerrors.uwerr(obs[i], wpm)
    end
end

ADerrors.uwerr(obs::FormalSeries.Series{ADerrors.uwreal,N}) where N =
    ADerrors.uwerr(obs, Dict{String,Vector{Float64}}())

import Base.:+
Base.:+(a::ADerrors.uwreal) = a

"""
    uwassign(obs, i; tagid="Ensemble")

Pull the `i`-th series coefficient out of a `Vector{Series{Float64,N}}`
across an ensemble and wrap it as a single `uwreal`.
"""
uwassign(obs::Vector{FormalSeries.Series{Float64,N}}, i; tagid="Ensemble") where {N} =
    ADerrors.uwreal([obs[k][i] for k in 1:length(obs)], tagid)