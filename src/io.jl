using Dates
using BSON: @save

"""
    git_commit_hash()

Return the current Git commit hash.
"""
function git_commit_hash()
    try
        return readchomp(`git rev-parse HEAD`)
    catch
        return "unknown"
    end
end


"""
    git_is_dirty()

Return `true` if the Git working tree contains uncommitted changes.
"""
function git_is_dirty()
    try
        return !isempty(readchomp(`git status --porcelain`))
    catch
        return missing
    end
end


"""
    build_run_metadata(config; data_file, run_tag)

Build metadata describing a training run.
"""
function build_run_metadata(
    config;
    data_file,
    run_tag,
)
    return Dict{String,Any}(
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "git_commit" => git_commit_hash(),
        "git_dirty" => git_is_dirty(),
        "seed" => config.seed,
        "run_tag" => run_tag,
        "data_file" => data_file,
        "config" => Dict(
            "kappa" => config.kappa,
            "lambda" => config.lambda,
            "epochs" => config.epochs,
            "batchsize" => config.batchsize,
            "test_batchsize" => config.test_batchsize,
            "nodes" => config.nodes,
            "split" => config.split,
            "lr" => config.lrate,
            "activation" => config.activation,
            "weight_decay" => config.weight_decay,
            "N_eval" => config.N_eval,
            "L1" => config.L1,
            "L2" => config.L2,
        ),
    )
end

"""
    save_training_results(...)

Save the trained model, metadata, and training diagnostics.
"""
function save_training_results(
    model,
    metadata,
    train_indices,
    test_indices,
    fval,
    ftest,
    wESS,
    ;
    model_path,
    metadata_path,
    plot=nothing,
    plot_path=nothing,
)
    @save model_path model

    @save metadata_path metadata train_indices test_indices fval ftest wESS

    if plot !== nothing && plot_path !== nothing
        savefig(plot, plot_path)
    end

    return nothing
end