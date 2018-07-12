include("laser.jl")
include("grid.jl")
using Laser
using Grid
using JLD
using Plots; gr()#GR#; pyplot()
using DifferentialEquations
using PyCall
using LaTeXStrings
@pyimport numpy as np

function transition_momentum(nE::Int, nx::Int, dx::Float64, dp::Float64, eMax::Float64, epsilon::Array{Float64,1}, u0::Array{Complex{Float64},2})
    D = zeros(Complex{Float64}, nE, nE)

    root_2eps = @. √(2epsilon)
    x, p = xpgrid(nx, dx, dp)
    @inbounds for jE in 1:nE
        for iE in 1:nE, j in 1:nx
            temp = zero(Complex{Float64})
            for i in eachindex(x)
                temp += cis(x[i] * root_2eps[iE] + x[j] * root_2eps[jE]) * (x[i] + x[j]) * conj(u0[i, j])
            end
            D[iE, jE] += temp
        end
        println("jE=", jE)
    end
    @. D *= dx^2 / 2π
    save("data/d.jld", "D", D)
    nothing
end

function plot_D()
    D = load("data/d.jld")["D"]::Array{Complex{Float64},2}
    absu = abs.(D)
    heatmap(absu)
    savefig("d.png")
    nothing
end

function evolution(du::Array{Complex{Float64},1}, u::Array{Complex{Float64},1}, p::Tuple{Array{Complex{Float64},2},Array{Float64,2},field,field,field,Float64}, t::Float64)
    # const V0 = 60.0/27.2114
    const V0 = 79.0 / 27.2114
    const d0 = 3.0
    # ele = GetEleField(t)
    ele = GetEleField(p[3], t) + GetEleField(p[4], t) #+ GetEleField(p[5], t)
    ele2 = GetEleField(p[5], t)
    temp = zero(Complex{Float64})
    shift = cis(-d0*ele2)
    for j in 1:nE, i in 1:nE
        temp += p[1][i, j] * shift * u[i + 1 + nE * (j - 1)]
        du[i + 1 + nE * (j - 1)] = -im * (conj(p[1][i, j] * shift) * ele * u[1] + (p[2][i, j] + V0) * u[i + 1 + nE * (j - 1)])
    end
    du[1] = -im * temp * ele * p[6]^2
    nothing
end

# function GetEleField(t_in::Float64)
#     const τ = 5e-15 / 2.4189e-17
#     const ω = 1.6#130/27.2114
#     const I_si = 8e17
#     const E_si = 2.742e3 * sqrt(I_si)
#     const E0 = E_si / 5.1421e11
#     t = t_in - 3 * τ
#     return exp(-t^2 / τ^2) * sin(ω * t) * E0
# end

function average(H::Array{Float64,1}, nx::Int)
    for i in 3:nx - 2
        H[i] = (H[i - 2] + H[i - 1] + H[i + 1] + H[i + 2]) / 4.0
    end
    H
end

function plot_ele()
    # const τ = 5e-15/2.4189e-17
    # const tmax = 6*τ
    # t = linspace(0, tmax, 1000)
    # ele = GetEleField.(t)
    plot(t, ele)
    savefig("ele.png")
end

function Kinetic(x::Float64, y::Float64)
    x + y
end


function energy_spectrum(u::Array{Float64,2}, epsilon::Array{Float64,1}, nE::Int64)
    E = repmat(epsilon, 1, nE)
    energy = map(Kinetic, E, E')
    energy = reshape(energy, nE * nE)

    absu = reshape(u, nE * nE)

    const nbins = 200
    y, bins = np.histogram(energy, nbins, range=[0, 1.0], weights=absu)
    y = y / maximum(y)
    for i in 1:2
        average(y, nbins)
    end
    y = y / maximum(y)
    # plot(bins[1:end - 1], y, xlabel = "E (a.u.)", ylabel = "Density")
    # plot(bins[1:end - 1], y, yscale = :log10, xlabel = "E (a.u.)", ylabel = "Density")
    plot(bins[1:end - 1], y, yscale=:log10, xlabel="E (a.u.)", ylabel="Density", xlims=[0, 1.0])
    savefig("e.png")
    nothing
end

function joint_spectrum(epsilon::Array{Float64,1})
    u0 = load("data/u.jld")["u"]
    u = reshape(abs.(u0[2:end]), (nE, nE))
    @time energy_spectrum(u, epsilon, nE)
    L = 0.5
    L2 = 1
    u ./= maximum(u)
    s1 = 1e-8
    s2 = 1
    for i in eachindex(u)
        u[i] = u[i] < s1 ? s1 :
            u[i] < s2 ? u[i] :
            s2
    end
    u ./= maximum(u)

    f = font(14, "Helvetica")
    heatmap(epsilon, epsilon, u,# nlevels = 50,
            xlabel=L"$\mathrm{E}_1 \mathrm{(a.u.)}$", ylabel=L"$\mathrm{E}_2 \mathrm{(a.u.)}$",
            titlefont=f, tickfont=f, legendfont=f, guidefont=f,
            aspect_ratio=1,
            # xticks=linspace(-L2, L2, 7), yticks=linspace(-L2, L2, 7),
            xlims=(0, L), ylims=(0, L),
            alpha=1.0,
            fillcolor=cgrad(:viridis, scale=:exp),
            colorbar=:right)
    savefig("je.png")
end

function dynamic()
    D = load("data/d.jld")["D"]::Array{Complex{Float64},2}
    u = zeros(Complex{Float64}, nE * nE + 1)
    esum = zeros(Float64, nE, nE)
    u[1] = 1.0

    # const τ = 5e-15/2.4189e-17
    for i in 1:nE, j in 1:nE
        esum[i, j] = epsilon[i] + epsilon[j]
    end
    de = eMax / nE
    prob = ODEProblem(evolution, u, (0.0, maximum(t)), (D, esum, f1, f2, f3, de))
    S = zeros(Complex{Float64}, nE, nE)

    @time sol = solve(prob, Tsit5(), save_everystep=false)

    save("data/u.jld", "u", sol.u[2])
end

const nE = 200
const eMax = 1.0
epsilon = Array(linspace(0, eMax, nE))
u0 = load("data/gs.jld")["u"]::Array{Complex{Float64},2}

# @time transition_momentum(nE, nx, dx, dp, eMax, epsilon, u0)
# @time plot_D()
# @time plot_ele()
@time dynamic()
@time joint_spectrum(epsilon)
