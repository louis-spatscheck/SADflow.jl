# scripts/plot_results.jl
#
# Plot results from a completed training run.
#
# This script does NOT train the model or recompute expensive quantities.
# It only loads the saved training results and generates diagnostic plots.
#
# Usage:
#   julia --project=. scripts/plot_results.jl \
#       --metadata results/models/<run>_metadata.jld2
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using JLD2
using Plots


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

function parse_commandline()

    args = ArgParseSettings()

    @add_arg_table args begin

        "--metadata"
            help = "Path to the metadata/results JLD2 file from train.jl"
            required = true
            arg_type = String

        "--output-dir"
            help = "Directory in which plots are saved"
            arg_type = String
            default = joinpath(@__DIR__, "..", "results", "plots")

        "--show"
            help = "Display plots interactively"
            action = :store_true
    end

    return parse_args(args)
end


# -----------------------------------------------------------------------------
# Plot training loss and ESS
# -----------------------------------------------------------------------------

function plot_training_results(
    fval,
    ftest,
    wESS;
    savepath,
)

    epochs = 1:length(fval)

    # Epochs for which a loss was actually evaluated.
    loss_epochs = findall(x -> x != 0.0, fval)

    # Epochs for which ESS was actually evaluated.
    ess_epochs = findall(x -> x != 0.0, wESS)

    isempty(loss_epochs) &&
        error("No training-loss data found.")

    isempty(ess_epochs) &&
        error("No ESS data found.")


    # -------------------------------------------------------------------------
    # Loss plot
    # -------------------------------------------------------------------------

    p_loss = plot(
        loss_epochs,
        fval[loss_epochs],
        label = "Train loss",
        linewidth = 2.5,
        xlabel = "Epoch",
        ylabel = "Loss",
        title = "Training and test loss",
        yscale = :log10,
        legend = :topright,
        grid = true,
        framestyle = :box,
    )

    plot!(
        p_loss,
        loss_epochs,
        ftest[loss_epochs],
        label = "Test loss",
        linewidth = 2.5,
    )


    # -------------------------------------------------------------------------
    # ESS plot
    # -------------------------------------------------------------------------

    p_ess = plot(
        ess_epochs,
        wESS[ess_epochs],
        label = "ESS",
        linewidth = 2.5,
        xlabel = "Epoch",
        ylabel = "Effective sample size",
        title = "Reweighting effective sample size",
        yscale = :log10,
        legend = :topright,
        grid = true,
        framestyle = :box,
    )


    # -------------------------------------------------------------------------
    # Combined figure
    # -------------------------------------------------------------------------

    p = plot(
        p_loss,
        p_ess,
        layout = (2, 1),
        size = (900, 750),
        dpi = 150,
        margin = 5Plots.mm,
    )

    savefig(p, savepath)

    return p
end


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()

    pargs = parse_commandline()

    metadata_file = pargs["metadata"]
    output_dir = pargs["output-dir"]

    mkpath(output_dir)

    # -------------------------------------------------------------------------
    # Load results
    # -------------------------------------------------------------------------

    @load metadata_file metadata train_indices test_indices fval ftest wESS

    println("Loaded training results:")
    println("  File:       $metadata_file")
    println("  Epochs:     $(length(fval))")
    println("  Train size: $(length(train_indices))")
    println("  Test size:  $(length(test_indices))")


    # -------------------------------------------------------------------------
    # Determine run name
    # -------------------------------------------------------------------------

    run_tag = get(metadata, "run_tag", "training_run")

    savepath = joinpath(
        output_dir,
        "$(run_tag)_training_results.png",
    )


    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    p = plot_training_results(
        fval,
        ftest,
        wESS;
        savepath = savepath,
    )

    println("Saved plot:")
    println("  $savepath")


    if pargs["show"]
        display(p)
    end

    return nothing
end


main()