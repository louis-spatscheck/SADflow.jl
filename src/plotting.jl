using Plots

function plot_training(train_loss, test_loss, ess, N; title="Training Overview")
    epochs = 1:2:N
    p = plot(epochs, train_loss[epochs] .- minimum(train_loss[1:N]) * 1.1,
             label="Train loss", color="#2166ac", linewidth=2.5,
             xlabel="Epoch", ylabel="Loss", title=title, yscale=:log10,
             legend=:topright, grid=true, gridalpha=0.3, gridstyle=:dash,
             framestyle=:box, size=(900, 450), dpi=150, margin=5Plots.mm)
    plot!(p, epochs, test_loss[epochs] .- minimum(test_loss[1:N]) * 1.1,
          label="Test loss", color="#d6604d", linewidth=2.5)

    p2 = twinx(p)
    plot!(p2, 1:5:N, 1 ./ ess[1:5:N], label="ESS", color="#4dac26",
          linewidth=2.5, yscale=:log10, linestyle=:dash, ylabel="ESS",
          legend=:bottomright, grid=false)
    return p
end

function build_heatmap_example(data::Array{T, Nd}, config; title="Heatmap", xlabel="X-axis", ylabel="Y-axis") where {T,Nd}
      example = data[:,:,1,1]  # Assuming data is of shape (Lx, Ly, C, N)
      p = heatmap(example, color=:viridis, xlabel=xlabel, ylabel=ylabel,
                title=title, size=(400, 350), dpi=150, margin=2Plots.mm)
      return p
end