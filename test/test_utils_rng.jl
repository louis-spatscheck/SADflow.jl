@testset "Batch sampling reproducibility" begin
    priors = reshape(Float64.(1:64), 4, 4, 1, 4)

    rng1 = MersenneTwister(1234)
    rng2 = MersenneTwister(1234)
    s1, inds1 = shuffle_data(rng1, priors)
    s2, inds2 = shuffle_data(rng2, priors)
    @test inds1[end] == inds2[end]
    @test s1 == s2

    rng3 = MersenneTwister(999)
    rng4 = MersenneTwister(999)
    b1, bi1 = select_random_batch(rng3, priors, 3)
    b2, bi2 = select_random_batch(rng4, priors, 3)
    @test bi1[end] == bi2[end]
    @test b1 == b2
end
