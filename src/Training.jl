# src/training.jl
#
# Training/evaluation orchestration helpers used by scripts/*.jl.
# Keep scientific kernels in losses.jl / observables.jl and keep scripts thin.

"""
    train_epoch!(model, opt, prior, batchsize, var_z, params; loss_function=KLloss_batch, K=20)

Run one training epoch over full shuffled data using contiguous batches.
"""
function train_epoch!(model, opt, prior, batchsize, var_z, params;
                      loss_function=KLloss_batch, K=20,
                      rng=Random.default_rng())
    shuffled_prior = shuffle_data(rng, prior)[1]
    iters = div(size(prior, 4), batchsize)

    for j in 1:iters
        start_idx = (j - 1) * batchsize + 1
        end_idx   = j * batchsize
        Xbatch = shuffled_prior[:, :, :, start_idx:end_idx]

        loss_val, grads = Flux.withgradient(model) do ml
            loss_function(Xbatch, ml, var_z, params; K=K)
        end
        Flux.update!(opt, model, grads[1])
    end

    return nothing
end

"""
    evaluate_losses(model, prior, prior_test, test_batchsize, var_z, params;
                    loss_function=KLloss_batch, K=25)

Evaluate train/test loss snapshots using randomly sampled batches.
"""
function evaluate_losses(model, prior, prior_test, test_batchsize, var_z, params;
                         loss_function=KLloss_batch, K=25,
                         rng=Random.default_rng())
    X_batch = select_random_batch(rng, prior, test_batchsize)[1]
    train_loss = loss_function(X_batch, model, var_z, params; K=K)

    X_test_batch = select_random_batch(rng, prior_test, test_batchsize)[1]
    test_loss = loss_function(X_test_batch, model, var_z, params; K=K)

    return train_loss, test_loss
end

"""
    evaluate_reweighting_checkpoint(model, prior_test, params;
                                    N=100, tag="rw", trace_samples=10)

Compute a periodic transformed-reweighting checkpoint summary.
Returns `(ess, corr, WT)`.
"""
function evaluate_reweighting_checkpoint(model, prior_test, params;
                                         N=100, tag="rw", trace_samples=10,
                                         rng=Random.default_rng())
    rw = reweighted_correlator(model, prior_test, params;
                               N=N, tag=tag, trace=true, ns_trace=trace_samples,
                               rng=rng)

    meanWT = ADerrors.uwreal(rw.WT, tag)
    ADerrors.uwerr(meanWT)
    ess = ADerrors.err.(meanWT.c[2])

    return (ess=ess, corr=rw.corr, WT=rw.WT)
end

"""
    TrainingConfig

Configuration for a LatticeFlow training run.
"""
struct TrainingConfig
    kappa::Float64
    lambda::Float64

    epochs::Int
    batchsize::Int
    test_batchsize::Int

    lrate::Float64
    split::Float64

    nodes::Int
    modes::Int
    activation::String

    seed::Int
    weight_decay::Float64
    N_eval::Int

    L1::Int
    L2::Int
    B::Int

    priors_dir::String
    output_dir::String
end