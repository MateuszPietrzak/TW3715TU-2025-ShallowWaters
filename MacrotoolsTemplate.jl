using LinearAlgebra
using BoundaryValueDiffEq
using SparseArrays
# using CairoMakie
using ModelingToolkit
using MacroTools: @capture, postwalk, prewalk

gamma::Float64 = 0.0
gamma_3::Float64 = 0.0
c::Float64 = 3

omega::Float64 = 2*pi*7

@parameters x,t
@variables A(..), B(..)
r1 = @rule cos(~x)^3 => 0.75 * cos(~x) + 0.25 * cos(3 * ~x)
r2 = @rule sin(~x)^3 => 0.75 * sin(~x) - 0.25 * sin(3 * ~x)
r3 = @rule cos(~x)^2 => 1 - sin(~x)^2
r4 = @rule sin(~x)^2 => 1 - cos(~x)^2


xleft::Float64 = 0.0
xright::Float64 = 1.0
N = 1000
order = 2
stepx = (xright-xleft)/N
u = A(x) * sin(omega*t) + B(x) * cos(omega*t)
Dx = Differential(x)
Dt = Differential(t)
y = Dt(Dt(u)) - c*c*Dx(Dx(u)) + gamma*Dt(u) + gamma_3*Dt(u)*Dt(u)*Dt(u)
y_exp = expand_derivatives(y)

y_exp = simplify(expand(y_exp), RuleSet([r1, r2, r3, r4]))
y_exp = expand(y_exp)
y_exp = simplify(y_exp, RuleSet([r1, r2, r3, r4]))

sin_coeff = -Symbolics.coeff(y_exp, sin(omega*t))

println("Sin coeff: ", sin_coeff)

function transform_sym(ex)
    return prewalk(ex) do x
        if @capture(x, Differential(x)(Differential(x)(s_(x))))
            return :(($s[i+1]-2* $s[i]+$s[i-1])/dx^2)
        elseif @capture(x, Differential(x)(s_(x)))
            return :(($s[i+1] - $s[i-1]) / (2*dx))
        elseif @capture(x, s_(x))
            return :($s[i])
        end
        return x
    end
end

function create_residual_function_1D(N, sin_eq, cos_expr)
    function_code = quote
        function residual!(F, U, p)
            dx = p

            A = @view(U[1:$N+1])
            B = @view(U[$N+2:end])
            F_A = @view(F[1:$N+1])
            F_B = @view(F[$N+2:end])

            # Inner points:
            for i in 2:$(N)
                F_A[i] = ($sin_eq) - 1000 * exp(-40(i * dx)^2)
                F_B[i] = ($cos_expr)
            end

            # Dirichlet BCs
            F_A[1] = A[1]
            F_A[end] = A[end]
            F_B[1] = B[1]
            F_B[end] = B[end]

            return F
        end
    end
    return eval(function_code)
end

println(toexpr(sin_coeff))
println(string(sin_coeff))

# This is the way to use it.
# That way its representation is correct
ex = Meta.parse(string(sin_coeff))
res = transform_sym(ex)
println(res)

# Wrong
res = transform_sym(toexpr(sin_coeff))
println(res)

# Wrong
res = transform_sym(sin_coeff)
println(res)
