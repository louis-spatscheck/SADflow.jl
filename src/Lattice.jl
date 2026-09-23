# src/Lattice.jl
#
# Lattice geometry, boundary-condition bookkeeping and nearest-neighbour
# helpers on periodic lattices.

const BC_PERIODIC = 0
const BC_SF_ORBI  = 1
const BC_SF_AFWB  = 2
const BC_OPEN     = 3

"""
    Grid{N,M,B,D}

Lattice geometry descriptor: `N`-dimensional lattice of size `iL`, blocked
into sub-blocks of size `blk`, with boundary condition `B` (one of
`BC_PERIODIC`, `BC_SF_ORBI`, `BC_SF_AFWB`, `BC_OPEN`) and twist angles `ntw`.

Two constructors:
- `Grid{N}(x, y)`            -> periodic BC, no twist
- `Grid{N}(x, y, ibc, nt)`   -> explicit BC + optional twist tuple `nt`
"""
struct Grid{N,M,B,D}
    ndim::Int64
    iL::NTuple{N,Int64}
    npls::Int64
    plidx::NTuple{M,Tuple{Int64,Int64}}

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

"""
    neighbour_sum(x)

Sum of the forward nearest neighbours `φ(x+t̂) + φ(x+x̂)` along the first
two (lattice) dims, with periodic boundary conditions. Works for fields of
any rank; the remaining dims (channel, batch) are not shifted.
"""


function neighbour_sum(x::AbstractArray{T,Nd}) where {T,Nd}
    shifts = (ntuple(i->(i==1 ? -1 : 0), Nd),
              ntuple(i->(i==2 ? -1 : 0), Nd))
    return circshift(x, shifts[1]) .+ circshift(x, shifts[2])
end

"""
    staple(f)

Sum of the four nearest-neighbour shifts of `f` (±x, ±y), used in the
φ⁴ action's hopping term.
"""

function staple(f::AbstractArray{T,Nd}) where {T,Nd}
    shifts = (ntuple(i->(i==1 ? -1 : 0), Nd),
              ntuple(i->(i==2 ? -1 : 0), Nd),
              ntuple(i->(i==1 ? 1 : 0), Nd),
              ntuple(i->(i==2 ? 1 : 0), Nd))
    fxp  = circshift(f, shifts[1])
    fyp  = circshift(f, shifts[2])
    fxp_ = circshift(f, shifts[3])
    fyp_ = circshift(f, shifts[4])
    return fxp .+ fyp .+ fxp_ .+ fyp_
    
end

"""
    pad_periodic(x, pads)

Pad the spatial dims of `x` (shape `(spatial..., C, B)`) with periodic
(circular) boundary conditions. `pads[d]` is either an `Int` (symmetric
pad) or a `(left, right)` tuple.
"""
function pad_periodic(x::AbstractArray, pads)
    N = ndims(x) - 2
    @assert length(pads) == N "pads must match number of spatial dims"

    out = x
    for d in 1:N
        p = pads[d]
        (pl, pr) = p isa Tuple ? p : (p, p)
        pl == 0 && pr == 0 && continue

        sz    = size(out, d)
        left  = selectdim(out, d, (sz - pl + 1):sz)
        right = selectdim(out, d, 1:pr)
        out   = cat(left, out, right; dims=d)
    end
    return out
end