module Laser

export field, gaussfield, trapzoidalfield, cos2field, GetEleField, ItoE0, nt, dt, t, ele, f1, f2, f3

abstract type field end

struct gaussfield <: field
    E0 :: Float64
    ω :: Float64
    ϕ :: Float64

    τ :: Float64
    tm :: Float64
end

struct trapzoidalfield <: field
    E0 :: Float64
    ω :: Float64
    ϕ :: Float64

    tStart :: Float64
    nUp :: Int
    nHori :: Int
    nDown :: Int
end

struct cos2field <: field
    E0 :: Float64
    ω :: Float64
    ϕ :: Float64

    τ  :: Float64
    tm :: Float64
end

function GetEleField(f::gaussfield, t_in::Float64)
    t = t_in - f.tm
    ele = f.E0 * cos(f.ω  * t - f.ϕ ) * exp(-2.0 * log(2.0) * (t / f.τ )^2)
end


function GetEleField(f::trapzoidalfield, t::Float64)
    ele = 0.0
    T = 2pi/f.ω
    t1 = f.tStart + f.nUp*T
    t2 = f.tStart + (f.nUp + f.nHori)*T
    t3 = f.tStart + (f.nUp + f.nHori + f.nDown)*T
    if (t > tStart && t < t1)
        ele = (t - tStart)/f.nUp/T
    end
    if (t > t1 && t < t2)
        ele = 1.0
    end
    if (t > t2 && t < t3)
        ele = (t - tStart - (f.nUp + f.nHori)*T)/(f.nDown)*T
    end

    ele = ele * f.E0 * cos(f.ω  * t - f.ϕ )
end

function GetEleField(f::cos2field, t::Float64)
    tStart = f.tm - f.τ /2
    tEnd = f.tm + f.τ /2
    ele = 0.0

    if (t > tStart && t < tEnd)
        ele = f.E0 * cos((t - f.tm) * pi / f.τ )^2 * cos(f.ω  * (t - f.tm) - f.ϕ )
    end
    ele
end

function ItoE0(I_si::Float64)
    E_si = 2.742e3 * sqrt(I_si)
    ItoE0 = E_si/5.1421e11
end

τ = 10e-15/2.4189e-17
f1 = cos2field(ItoE0(1.0e15), 3.2, 0.0, τ, (6-0.5)τ/2)
f2 = cos2field(ItoE0(0.0e16), 1.66, 0.0, τ, (6+0.25)τ/2)
f3 = cos2field(ItoE0(1.0e15), 0.04, 0.0, 6τ, 6τ/2)

const tExtra = 100./0.3
const dt = 0.1
const nt = round(Int, (max(f1.τ/2 + f1.tm, f2.τ/2 + f2.tm, f3.τ/2 + f3.tm) + tExtra)/dt)
t = linspace(0, nt, nt)*dt
ele = zeros(nt)

@. ele = GetEleField(f1, t) + GetEleField(f2, t) + GetEleField(f3, t)

end
