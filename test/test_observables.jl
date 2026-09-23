@testset "Observable smoke" begin
    rng  = MersenneTwister(7)
    pics = rand(rng, Float32, 4, 4, 1, 6)
    corr_fft = two_point_corr_fft(pics)
    @test size(corr_fft) == (4, 4)
    @test eltype(corr_fft) == Float32
    @test all(isfinite, corr_fft)

    # The Γ-method error analysis in ADerrors needs a Monte Carlo history of
    # reasonable length; with only a handful of measurements its automatic
    # windowing indexes past the end of the autocorrelation function.
    ncfg = 500
    cfgs = randn(rng, Float64, 4, 4, 1, ncfg)
    corr = two_point_correlator(cfgs, 4; ncfg=ncfg, tag="test")
    @test length(corr) == 4
end
