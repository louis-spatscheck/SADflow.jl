@testset "Model, loss, and training smoke" begin
    @test activation_fn("tanh") === tanh
    @test_throws Exception activation_fn("does_not_exist")
    L1, L2, B = 4, 4, 4
    space = Grid{2}((L1, L2), (B, B))
    params_phi4 = Phi4Params(0.24, 0.0)

    model = make_model(space, params_phi4 ; nodes=2, activation=tanh)
    x = rand(Float64, L1, L2, 1, 8)
    y = model(x)
    @test size(y) == size(x)


    var_z = 1.0
    loss = KLloss_batch(x, model, var_z, params_phi4; K=1)
    @test isfinite(loss)

    opt = Flux.setup(Flux.Adam(1e-3), model)
    rng = MersenneTwister(5)
    @test train_epoch!(model, opt, x, 2, var_z, params_phi4; K=1, rng=rng) === nothing

    train_loss, test_loss = evaluate_losses(model, x, x, 4, var_z, params_phi4; K=1, rng=rng)
    @test isfinite(train_loss)
    @test isfinite(test_loss)
end
