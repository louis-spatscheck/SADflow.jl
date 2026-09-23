using Test
using Random
using Flux

include(joinpath(@__DIR__, "..", "src", "SADflow.jl"))
using .SADflow

include("test_api.jl")
include("test_lattice_action.jl")
include("test_utils_rng.jl")
include("test_models_losses_training.jl")
include("test_observables.jl")
