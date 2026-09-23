# scripts/train.jl
#
# Entry point: parses CLI args, loads prior configs, builds the model,
# runs the training loop, and saves the trained model + a quick training-
# curve diagnostic plot. All physics/model/loss logic lives in src/ — this
# script is orchestration only. Source: training_5.jl (whole file), with
# the shared logic pulled into SADflow and only the script-specific
# bits (CLI parsing, data loading, the epoch loop, saving) left here.
#
# Usage:
#   julia --project=. scripts/train.jl -k 0.24 -l 0.0 -e 200 -b 64 -n 4 -a tanh

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SADflow
using ADerrors, FormalSeries
using Flux
using ArgParse
using Random
using CUDA

# --- CLI -----------------------------------------------------------------

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
        "--lr", "-r"
            arg_type = Float64
            default = 1e-3
        "--split"
            arg_type = Float64
            default = 0.75
        "-m", "--modes"
            arg_type = Int
            default = 1
        "-n", "--nodes"
            arg_type = Int
            default = 4
        "-a", "--activation"
            arg_type = String
            default = "tanh"
        "--priors-dir"
            arg_type = String
            default = joinpath(@__DIR__, "..", "priors")
        "--output-dir"
            arg_type = String
            default = joinpath(@__DIR__, "..", "results")
        "--seed"
            arg_type = Int
            default = 1234
    end
    return parse_args(args)
end

pargs = parse_commandline()



config = TrainingConfig(
    pargs["kappa"],
    pargs["lambda"],
    pargs["epochs"],
    pargs["batchsize"],
    pargs["test_batchsize"],
    pargs["lr"],
    pargs["split"],
    pargs["nodes"],
    pargs["modes"],
    pargs["activation"],
    pargs["seed"],
    0.0,       # weight_decay
    100,       # N_eval
    8,         # L1
    8,         # L2
    8,         # B
    pargs["priors-dir"],
    pargs["output-dir"],
)

Random.seed!(config.seed)
rng = MersenneTwister(config.seed)


USE_GPU = CUDA.functional() 
to_device(x) = USE_GPU ? CUDA.cu(x) : x


space = Grid{2}((8, 8 ), (8, 8))
params_phi4 = Phi4Params(config.kappa, config.lambda)

N_eval = 100




plots_dir  = joinpath(config.output_dir, "plots")
models_dir = joinpath(config.output_dir, "models")
mkpath(plots_dir); mkpath(models_dir)

run_tag = "2d_l$(config.lambda)_k$(config.kappa)_L_$(config.L1)_$(config.L2)_nodes$(config.nodes)_bs$(config.batchsize)_activation$(config.activation)_lr$(config.lrate)"

model_path = joinpath(models_dir, "$(run_tag).bson")
metadata_path = joinpath(models_dir, "$(run_tag)_metadata.jld2")



# --- Load priors -----------------------------------------------------------

data_file = joinpath(
    config.priors_dir,
    "2d_l$(config.lambda)_k$(config.kappa)_L_$(config.L1)_$(config.L2).jld2",
)

metadata = build_run_metadata(
    config;
    data_file=data_file,
    run_tag=run_tag,   
)

pics = load_prior_data(data_file)

var_z = estimate_source_variance(pics)

prior, prior_test, train_indices, test_indices =
    to_device.(split_data(pics, config.split; rng=rng))

# --- Model + optimiser -----------------------------------------------------

activ = activation_fn(config.activation)
model = to_device(make_model(
    space, 
    params_phi4; 
    nodes=config.nodes, 
    activation=activ))

loss_function = KLloss_batch

optimiser = OptimiserChain(WeightDecay(config.weight_decay), Flux.Adam(config.lrate))
opt = Flux.setup(optimiser, model)

# --- Training loop -----------------------------------------------------

fval = zeros(Float64, config.epochs)
ftest = zeros(Float64, config.epochs)
wESS  = zeros(Float64, config.epochs)
correlators_trw = Vector{Vector{ADerrors.uwreal}}(undef, config.epochs)

for epoch in 1:config.epochs
    train_epoch!(model, opt, prior, config.batchsize, var_z, params_phi4;
                 loss_function=loss_function, K=10, rng=rng)

    if epoch % 2 == 1
        train_loss, test_loss = evaluate_losses(model, prior, prior_test, config.test_batchsize,
                                                var_z, params_phi4;
                                                loss_function=loss_function, K=10,
                                                rng=rng)
        fval[epoch] = train_loss
        ftest[epoch] = test_loss

        println("Epoch $epoch | Train Loss: $(fval[epoch])")
        println("Epoch $epoch | Test Loss:  $(ftest[epoch])")
    end

    if epoch % 5 == 1
        rw = evaluate_reweighting_checkpoint(model, prior_test, params_phi4;
                                             N=config.N_eval, tag=string(epoch), trace_samples=10,
                                             rng=rng)
        wESS[epoch] = rw.ess
        correlators_trw[epoch] = rw.corr
        println("Epoch $epoch | Test ESS: $(rw.ess)")
    end
end

# --- Save results -----------------------------------------------------

ep_end = config.epochs - 1
p = plot_training(fval, ftest, wESS, ep_end)

save_training_results(
    model,
    metadata,
    train_indices,
    test_indices,
    fval,
    ftest,
    wESS;
    model_path=model_path,
    metadata_path=metadata_path,
    plot=p,
    plot_path=joinpath(plots_dir, "$(run_tag)_training.png"),
)

println("Done. Model saved to $(joinpath(models_dir, "$(run_tag).bson"))")