@testset "Observable smoke" begin
    pics = rand(Float32, 4, 4, 1, 6)
    corr_fft = two_point_corr_fft(pics)
    @test size(corr_fft) == (4, 4)
    @test eltype(corr_fft) == Float32
    @test all(isfinite, corr_fft)

    corr = two_point_correlator(Float64.(pics), 4; ncfg=6, tag="test")
    @test length(corr) == 4
end
