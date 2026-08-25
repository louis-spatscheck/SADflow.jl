@testset "Lattice and action invariants" begin
    κ = 0.24
    λ = 0.5
    params = Phi4Params(κ, λ)

    field = reshape(Float64.(1:16), 4, 4, 1, 1)
    @test size(neighbour_sum(field)) == size(field)
    @test size(staple(field)) == size(field)

    shifted = circshift(field, (1, 2, 0, 0))
    @test action(shifted, params) == action(field, params)

    constant_field = ones(Float64, 4, 4, 1, 1)
    expected = 4 * 4 * (-4κ + 1)
    @test only(action(constant_field, params)) ≈ expected
end
