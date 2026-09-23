# src/Utils.jl
#
# Batch sampling helpers used in the training loop.

"""
    shuffle_data(priors)

Shuffle `priors` along its last dimension (the batch/config dim). Returns
`(shuffled_array, index_tuple)`.
"""
function shuffle_data(rng::AbstractRNG, priors::AbstractArray)
    d = ndims(priors)
    N = size(priors, d)
    shuffled_indices = randperm(rng, N)
    inds = ntuple(i -> (i == d ? shuffled_indices : :), d)
    return priors[inds...], inds
end

shuffle_data(priors::AbstractArray) = shuffle_data(Random.default_rng(), priors)

"""
    select_random_batch(priors, batch_size)

Draw `batch_size` random samples (with replacement) along the last
dimension of `priors`. Returns `(batch_array, index_tuple)`.
"""
function select_random_batch(rng::AbstractRNG, priors::AbstractArray, batch_size::Int)
    d = ndims(priors)
    d < 1 && error("select_random_batch: priors must have at least 1 dimension")
    N = size(priors, d)
    selected_indices = rand(rng, 1:N, batch_size)
    inds = ntuple(i -> (i == d ? selected_indices : :), d)
    return priors[inds...], inds
end

select_random_batch(priors::AbstractArray, batch_size::Int) =
    select_random_batch(Random.default_rng(), priors, batch_size)