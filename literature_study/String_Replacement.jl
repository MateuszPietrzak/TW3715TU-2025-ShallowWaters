using LinearAlgebra
using BoundaryValueDiffEq
using SparseArrays
using Plots
using ModelingToolkit
using MethodOfLines
using DifferentialEquations
using NLsolve, NonlinearSolve
using DomainSets

xleft::Float64 = 0.0
xright::Float64 = 1.0
yleft::Float64 = 0.0
yright::Float64 = 1.0
N = 100
order = 2
stepx = (xright-xleft)/N
stepy = (yright-yleft)/N

#Definition of constants (placeholders):
gamma::Float64 = 1
gamma_3::Float64 = 1e-1
c::Float64 = 1
omega::Float64 = 1

@parameters x, y, t
@variables A(..), B(..)
r1 = @rule cos(~x)^3 => 0.75 * cos(~x) + 0.25 * cos(3 * ~x)
r2 = @rule sin(~x)^3 => 0.75 * sin(~x) - 0.25 * sin(3 * ~x)
r3 = @rule cos(~x)^2 => 1 - sin(~x)^2
r4 = @rule sin(~x)^2 => 1 - cos(~x)^2

u = A(x, y) * sin(omega*t) + B(x, y) * cos(omega*t)
Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)
y_eq = Dt(Dt(u)) - c*c*(Dx(Dx(u)) + Dy(Dy(u))) + gamma*Dt(u) + gamma_3*Dt(u)*Dt(u)*Dt(u)
y_exp = expand_derivatives(y_eq)

y_exp = simplify(expand(y_exp), RuleSet([r1, r2, r3, r4]))
y_exp = expand(y_exp)
y_exp = simplify(y_exp, RuleSet([r1, r2, r3, r4]))

sin_coeff = Symbolics.coeff(y_exp, sin(omega*t))
cos_coeff = Symbolics.coeff(y_exp, cos(omega*t))
#println("Sin coefficient equation:")
#println(sin_coeff)
#println("Cos coefficient equation:")
#println(cos_coeff)


function transform_sym(expr_str)
    corrected = string(expr_str)

    repl_lapl = [
        "Differential(x)(Differential(x)(A(x, y)))" => "(A[i+1, j] - 2A[i, j] + A[i-1, j]) / dx^2",
        "Differential(y)(Differential(y)(A(x, y)))" => "(A[i, j+1] - 2A[i, j] + A[i, j-1]) / dy^2",
        "Differential(x)(Differential(x)(B(x, y)))" => "(B[i+1, j] - 2B[i, j] + B[i-1, j]) / dx^2",
        "Differential(y)(Differential(y)(B(x, y)))" => "(B[i, j+1] - 2B[i, j] + B[i, j-1]) / dy^2",
        ]
    repl_first_der = [
        "Differential(x)(A(x, y))" => "(A[i+1, j] - A[i-1, j]) / (2*dx)",
        "Differential(y)(A(x, y))" => "(A[i, j+1] - A[i, j-1]) / (2*dy)",
        "Differential(x)(B(x, y))" => "(B[i+1, j] - B[i-1, j]) / (2*dx)",
        "Differential(y)(B(x, y))" => "(B[i, j+1] - B[i, j-1]) / (2*dy)",
        ]
    repl_linear = [
        "A(x, y)" => "A[i, j]",
        "B(x, y)" => "B[i, j]"
        ]

    corrected = replace(corrected, repl_lapl...)
    corrected = replace(corrected, repl_first_der...)
    corrected = replace(corrected, repl_linear...)

    repl_powers = [
        "A[i, j]^" => "(A[i, j])^",
        "B[i, j]^" => "(B[i, j])^"
        ]
    repl_mult_pattern = r"(?<=\d)(?=[A-Za-z\(])"

    corrected = replace(corrected, repl_powers...)
    corrected = replace(corrected, repl_mult_pattern => "*")

    #Finar corrections (required for scope):
    repl_sym = [
        "A" => "A_array",
        "B" => "B_array",
        "i" => "i_local",
        "j" => "j_local"
        ]

    corrected = replace(corrected, repl_sym...)

    return corrected
end


println("Trial substitution:")
println(transform_sym(sin_coeff))

println("Trial substitution:")
println(transform_sym(cos_coeff))


function create_residual_function(N, sin_eq_str, cos_expr_str)
    function_code = """
    function residual!(F, U, p)
    dx, dy = p

    A_array = reshape(@view(U[1:$N*$N]), $N, $N)
    B_array = reshape(@view(U[$N*$N+1:end]), $N, $N)
    F_A = reshape(@view(F[1:$N*$N]), $N, $N)
    F_B = reshape(@view(F[$N*$N+1:end]), $N, $N)

    # Inner points:
    for i_local in 2:$(N-1)
        for j_local in 2:$(N-1)
            F_A[i_local, j_local] = ($sin_eq_str) - 10*sin(i_local*dx + j_local*dy)
            F_B[i_local, j_local] = ($cos_expr_str) - 10*cos(i_local*dx + j_local*dy)
        end
    end

    # BCs (Dirichlet for now):
        F_A[1,:] .= A_array[1,:]; F_A[end,:] .= A_array[end,:]
        F_A[:,1] .= A_array[:,1]; F_A[:,end] .= A_array[:,end]
        F_B[1,:] .= B_array[1,:]; F_B[end,:] .= B_array[end,:]
        F_B[:,1] .= B_array[:,1]; F_B[:,end] .= B_array[:,end]

        return F
    end
    """

    return eval(Meta.parse(function_code))
end

sin_eq_str = transform_sym(sin_coeff)
cos_eq_str = transform_sym(cos_coeff)

nonlinear_residual = create_residual_function(N, sin_eq_str, cos_eq_str)
initial_guess = ones(2*N^2)
par = [stepx, stepy]

prob = NonlinearProblem(nonlinear_residual, initial_guess, par)
sol = NonlinearSolve.solve(prob, NewtonRaphson(), reltol = 1e-5, abstol = 1e-5)

#print(sol.u)


A_sol = reshape(sol.u[1:N*N], N, N)
B_sol = reshape(sol.u[N*N+1:end], N, N)


xgrid = range(xleft, xright, length=N)
ygrid = range(yleft, yright, length=N)


tgrid = 0.0:0.1:30.0
total_frames = length(tgrid)


anim = @animate for t in tgrid
    u_t = A_sol .* sin(omega*t) .+ B_sol .* cos(omega*t)
    heatmap(xgrid, ygrid, u_t',
            color=:viridis,
            xlabel="x", ylabel="y",
            title="u(x,y,t) at t=$t) s",
            clims=(-1, 1),
            aspect_ratio=1)
end

gif(anim, "wave_equation_30s_asymm.gif", fps=25)
