@testset "API compatibility" begin
    @test isdefined(LamPhi4_trw, :Phi4Params)
    @test isdefined(LamPhi4_trw, :action)
    @test isdefined(LamPhi4_trw, :make_model)
    @test isdefined(LamPhi4_trw, :train_epoch!)

    p_old = Phi4Params(0.24, 0.0)
    @test p_old.κ == 0.24
    @test p_old.λ == 0.0

    if isdefined(LamPhi4_trw, :Phi4Params)
        p_new = LamPhi4_trw.Phi4Params(0.24, 0.0)
        @test p_new.κ == p_old.κ
        @test p_new.λ == p_old.λ
    end

    x = rand(Float64, 2, 2, 1, 1)
    @test action(x, p_old) isa AbstractArray

    if isdefined(LamPhi4_trw, :action)
        @test LamPhi4_trw.action(x, p_old) == action(x, p_old)
    end
end
