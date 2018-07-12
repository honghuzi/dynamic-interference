module Grid
export nx, dx, dp, sg,
    xpgrid

struct spacegrid
    nx :: Int
    dx :: Float64
    dp :: Float64
    xmax :: Float64
    pmax :: Float64
    La :: Float64
    Ls :: Float64
    δ :: Float64
end

function xpgrid(nx::Int , dx::Float64, dp::Float64)
    x = Array((-nx/2+1 : nx/2) * dx)
    p = Array((-nx/2+1 : nx/2) * dp)
    p = circshift(p, round(Int, nx/2))
    x, p
end

const nx = 2^12
const dx = 0.2
const dp = 2pi / (nx * dx)
const xmax = dx*nx/2
const pmax = pi / dx
const La = xmax - 30.0
const Ls = 150.0
const δ = 10.0

sg = spacegrid(nx, dx, dp, xmax, pmax, La, Ls, δ)

end
