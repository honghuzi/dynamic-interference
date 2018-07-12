include("laser.jl")
include("grid.jl")
using Laser
using Grid
using JLD
# using Plots; gr()#GR#; pyplot()
using DifferentialEquations
using PyCall
# using LaTeXStrings
@pyimport numpy as np

function transition_momentum(nE::Int, nx::Int, dx::Float64, dp::Float64, eMax::Float64, epsilon::Array{Float64,1}, u0::Array{Complex{Float64},2}, u1::Array{Complex{Float64}, 1})
    D = zeros(Complex{Float64}, nE)

    root_2eps = @. √(2epsilon)
    x, p = xpgrid(nx, dx, dp)
    @inbounds for jE in 1:nE
        for j in 1:nx
            temp = zero(Complex{Float64})
            for i in eachindex(x)
                temp += cis(x[j] * root_2eps[jE]) * u1[i] * (x[i] + x[j]) * conj(u0[i, j])
            end
            D[jE] += temp
        end
        println("jE=", jE)
    end
    @. D *= dx^2 / √(2π)
    save("data/d.jld", "D", D)
    nothing
end

function transition_momentum_Ion(nE::Int, nx::Int, dx::Float64, dp::Float64, eMax::Float64, epsilon::Array{Float64,1}, u0::Array{Complex{Float64},1})
    D = zeros(Complex{Float64}, nE)

    root_2eps = @. √(2epsilon)
    x, p = xpgrid(nx, dx, dp)
    @inbounds for iE in 1:nE
        for i in eachindex(x)
            D[iE] += cis(x[i] * root_2eps[iE]) * x[i] * conj(u0[i])
        end
    end
    @. D *= dx / √(2π)
    save("data/dIon.jld", "D", D)
    nothing
end


function plot_D()
    D = load("data/d.jld")["D"]::Array{Complex{Float64},2}
    absu = abs.(D)
    heatmap(absu)
    savefig("d.png")
    nothing
end

## D, DIon, epsilon, esum, f1, f2, f3, de, Γ_N, Δ_I, Δ_I2
function evolution(du::Array{Complex{Float64},1}, u::Array{Complex{Float64},1},
                   p::Tuple{Array{Complex{Float64},1}, ##p1 : D
                            Array{Complex{Float64},1}, ##p2 : DIon
                            Array{Float64,1},          ##p3 : epsilon
                            Array{Float64,2},          ##p4 : esum
                            field,field,field,         ##p5, p6, p7 : f1, f2, f3
                            Float64,                   ##p8 : de
                            Float64,                   ##p9 : Γ_N
                            Float64,                   ##p10 : Δ_I
                            Float64,                   ##p11 : Δ_I2
                            },
                   t::Float64)
    # const V0 = 60.0/27.2114
    const V2 = 2.89384 #2.913
    const V1 = 0.89384 #V2 - 2.0024
    D = @. p[1]
    DIon = @. p[2]
    epsilon = @. p[3]
    esum = @. p[4]

    g1 = GetEnv(p[5], t) 
    # g12 = GetEnv(p[6], t)
    g1sq = g1^2 # + g12^2
    # g1 = g1 + g12
    g2 = GetEnv(p[7], t)
    g2sq = g2^2
    de = p[8]
    Γ_N = p[9]
    Δ_I = p[10]
    Δ_I2 = p[11]
    ω = p[5].ω
    E0 = p[5].E0

    du[1] = -im * 0.5 * Γ_N * g1sq * u[1]
    for j in 1:nE
        du[1+j] = 0.5 * D[j] * E0 * g1 * u[1] + (V1 + Δ_I * g1sq + Δ_I2 * g2sq + epsilon[j] - ω) * u[1+j]
        for i in 1:nE
            du[i + 1 + nE * j] = 0.5 * DIon[i] * E0 * g1 * u[1+j] + (V2 + esum[i, j] - 2ω) * u[i + 1 + nE * j]
        end
    end
    @. du *= -im
    nothing
end

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

    save("data/esum.jld", "e", bins[1:end-1], "y", y)
    nothing
end

function dynamic()
    D = load("data/d.jld")["D"]::Array{Complex{Float64},1}
    DIon = load("data/dIon.jld")["D"]::Array{Complex{Float64},1}
    u = zeros(Complex{Float64}, (nE + 1) * nE + 1)
    esum = zeros(Float64, nE, nE)
    u[1] = 1.0

    for i in 1:nE, j in 1:nE
        esum[i, j] = epsilon[i] + epsilon[j]
    end
    de = eMax / nE
    const Γ_N =  0.01 #/27.2114  #2e-3
    const Δ_I = 0.1    #/27.2114  #-6e-3
    const Δ_I2 = 0.5   #3#.1#10/27.2114          #-6e-2


    prob = ODEProblem(evolution, u, (0.0, maximum(t)), (D, DIon, epsilon, esum, f1, f2, f3, de, Γ_N, Δ_I, Δ_I2))
    S = zeros(Complex{Float64}, nE, nE)

    @time sol = solve(prob, Tsit5(), save_everystep=false)

    uall = sol.u[2]
    ug = uall[1]
    usi = uall[2 : 1+nE]
    udi = uall[2+nE : end]
    udi = reshape(udi, (nE, nE))
    # energy_spectrum(abs2.(udi), epsilon, nE)
    save("data/u.jld", "u", uall)
    save("data/si.jld", "si", abs2.(usi)/maximum(abs2.(usi)), "e", epsilon)
    save("data/di.jld", "di", (abs2.(udi)+abs2.(udi'))/maximum(abs2.(udi)))
end

const nE = 200
const eMax = 1.0
epsilon = Array(linspace(0, eMax, nE))
u0 = load("data/gs.jld")["u"]::Array{Complex{Float64},2}
uIon = load("data/gsIon.jld")["u"]::Array{Complex{Float64},1}
u0 = conj.(u0)
uIon = conj.(uIon)
# @time transition_momentum(nE, nx, dx, dp, eMax, epsilon, u0, uIon)
# @time transition_momentum_Ion(nE, nx, dx, dp, eMax, epsilon, uIon)
# @time plot_D()
# @time plot_ele()
@time dynamic()
# @time joint_spectrum(epsilon)