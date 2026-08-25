# src/data.jl



"""
    load_prior_data(path; first_config=1000, last_config=2000, T=Float64)

Load prior φ⁴ configurations from a JLD2 file.

Returns the selected configurations as an `Array{T,4}`.
"""
function load_prior_data(
    path;
    first_config=1000,
    last_config=2000,
    T=Float64,
)
    @load path pics

    return convert(
        Array{T,4},
        pics[:, :, :, first_config:last_config],
    )
end

"""
    split_data(data, split; rng=Random.default_rng())

Split configurations into training and test sets.

Returns `(train_data, test_data, train_indices, test_indices)`.
"""
function split_data(data, split; rng=Random.default_rng())
    N = size(data, 4)

    ntrain = floor(Int, split * N)

    indices = randperm(rng, N)
    train_indices = indices[1:ntrain]
    test_indices = indices[ntrain+1:end]

    train_data = data[:, :, :, train_indices]
    test_data = data[:, :, :, test_indices]

    return train_data, test_data, train_indices, test_indices
end