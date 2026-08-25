module LamPhi4_trw


using ADerrors
using FormalSeries
using Flux
using Functors
using ForwardDiff
using Zygote
using Random
using Statistics
using LinearAlgebra
using JLD2: @load
import AbstractFFTs: fft, ifft
import ForwardDiff: Dual, partials, value, Partials

# Order matters: each file only depends on ones included before it.
include("Lattice.jl")       # Grid, boundary conditions, staple, neighbour_sum
include("AutoDiff.jl")      # Dual-aware fft/ifft, ADerrors <-> FormalSeries glue
include("ActionPhi4.jl")        # Phi4_params, action, f_HS_f
include("Models.jl")        # PeriodicConv, EffectivePropagator, make_model1
include("Losses.jl")        # trJ/trJJ estimators, KLloss family
include("Utils.jl")    # shuffle_data, select_random_batch
include("Observables.jl")   # correlators, effective mass, reweighting
include("Training.jl")      # train/eval orchestration helpers
include("data.jl")
include("plotting.jl")      # plotting helpers
include("io.jl")            # save/load helpers, run metadata
# --- Public API -------------------------------------------------------
# Keep this list in sync with what scripts/*.jl actually use. Anything not
# exported here can still be reached as LatticeFlow.foo, which is fine for
# the more "internal" helpers (e.g. stack_complex_flux, build_source_flux).

export Grid, BC_PERIODIC, BC_SF_ORBI, BC_SF_AFWB, BC_OPEN
export staple, neighbour_sum, pad_periodic


export Phi4Params, action, hessian, source_derivative

export PeriodicConv, EffectivePropagator, make_model, make_CNN, ModelWrapper, pool_field, activation_fn

export trJ, trJJ, KLloss, KLloss_batch, l2_penalty

export shuffle_data, select_random_batch

export two_point_correlator, two_point_corr_fft, correlator,
       reweighted_correlator, effective_mass_cosh, effective_mass_acosh,
       correlator_variance, estimate_source_variance

export train_epoch!, evaluate_losses, evaluate_reweighting_checkpoint, TrainingConfig
export load_prior_data, split_data
export save_training_results,build_run_metadata, git_commit_hash, git_is_dirty
export plot_training, build_heatmap_example   
end # module