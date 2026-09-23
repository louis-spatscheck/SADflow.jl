@testset "API compatibility" begin
    @test isdefined(SADflow, :Phi4Params)
    @test isdefined(SADflow, :action)
    @test isdefined(SADflow, :make_model)
    @test isdefined(SADflow, :train_epoch!)

    p_old = Phi4Params(0.24, 0.0)
    @test p_old.κ == 0.24
    @test p_old.λ == 0.0

    if isdefined(SADflow, :Phi4Params)
        p_new = SADflow.Phi4Params(0.24, 0.0)
        @test p_new.κ == p_old.κ
        @test p_new.λ == p_old.λ
    end

    x = rand(Float64, 2, 2, 1, 1)
    @test action(x, p_old) isa AbstractArray

    if isdefined(SADflow, :action)
        @test SADflow.action(x, p_old) == action(x, p_old)
    end
end
