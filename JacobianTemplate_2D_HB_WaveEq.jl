using LinearAlgebra
using SparseArrays
using ModelingToolkit
using MacroTools: @capture, postwalk, prewalk
using DifferentialEquations
using NonlinearSolve
using Plots

xleft::Float64 = 0.0
xright::Float64 = 1.0
yleft::Float64 = 0.0
yright::Float64 = 1.0
N = 125
stepx = (xright-xleft)/(N-1)
stepy = (yright-yleft)/(N-1)

#Definition of constants (placeholders):
gamma::Float64 = 10.0
gamma_3::Float64 = 1.0
c::Float64 = 10.0
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

#println("Sin coeff: ", sin_coeff)

function transform_sym(ex)
    return prewalk(ex) do instr
        if @capture(instr, Differential(x)(Differential(x)(s_(x, y))))
            return :(($s[i+1, j]-2* $s[i, j]+$s[i-1, j])/dx^2)
            elseif @capture(instr, Differential(y)(Differential(y)(s_(x, y))))
            return :(($s[i, j+1]-2* $s[i, j]+$s[i, j-1])/dy^2)
            elseif @capture(instr, Differential(x)(s_(x, y)))
            return :(($s[i+1, j] - $s[i-1, j]) / (2*dx))
            elseif @capture(instr, Differential(y)(s_(x, y)))
            return :(($s[i, j+1] - $s[i, j-1]) / (2*dy))
            elseif @capture(instr, s_(x, y))
            return :($(Symbol("$(s)_array"))[i, j])
            elseif @capture(instr, s_[i_, j_])
            return :($(Symbol("$(s)_array"))[$(i), $(j)])
        end
        return instr
    end
end

function transform_sym_coeff(ex)
    return prewalk(ex) do instr
        if @capture(instr, s_(x, y))
            return :($(Symbol("$(s)_array"))[idx])
        end
        return instr
    end
end

function create_residual_function(N, sin_eq, cos_expr)
    function_code = quote
        function residual!(F, U, p)
            dx, dy = p
            grid_size = $N * $N

            A_array = reshape(@view(U[1:grid_size]), $N, $N)
            B_array = reshape(@view(U[grid_size+1:2*grid_size]), $N, $N)


            F_A = reshape(@view(F[1:grid_size]), $N, $N)
            F_B = reshape(@view(F[grid_size+1:2*grid_size]), $N, $N)


            # Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    F_A[i, j] = ($sin_eq) - 1000 * c^2 * exp(-40*(i * dx)^2)
                    F_B[i, j] = ($cos_expr)
                end
            end

            # BCs (Dirichlet for now):
            F_A[1,:] .= A_array[1,:]; F_A[end,:] .= A_array[end,:]
            F_A[:,1] .= A_array[:,1]; F_A[:,end] .= A_array[:,end]
            F_B[1,:] .= B_array[1,:]; F_B[end,:] .= B_array[end,:]
            F_B[:,1] .= B_array[:,1]; F_B[:,end] .= B_array[:,end]


            return F
        end
    end
    return eval(function_code)
end

function create_jacobian_function(N, DiffMat, LaplCoeff)
    function_code = quote
        function jacobian!(J, U, p)
            dx, dy = p
            grid_size = $N * $N

            A_array = @view(U[1:grid_size])
            B_array = @view(U[grid_size+1:2*grid_size])

            Jmat = spzeros(2*grid_size, 2*grid_size)

            #Boundary conditions (easier to write over all diagonal points, then overwrite later):
            for k in 1:2*grid_size
                Jmat[k,k] = 1
            end

            #Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    idx = (j-1)*$N + i
                    Jmat[idx, idx] = $(DiffMat[1][1]) - 2*LaplCoeff[1][1]/(dx^2) - 2*LaplCoeff[1][2]/(dy^2)
                    Jmat[idx, idx  + grid_size] = $(DiffMat[1][2])
                    Jmat[idx + grid_size, idx] = $(DiffMat[2][1])
                    Jmat[idx + grid_size, idx + grid_size] = $(DiffMat[2][2]) - 2*LaplCoeff[2][1]/(dx^2) - 2*LaplCoeff[2][2]/(dy^2)

                    Jmat[idx, idx - 1] = LaplCoeff[1][1]/(dx^2)
                    Jmat[idx, idx + 1] = LaplCoeff[1][1]/(dx^2)
                    Jmat[idx, idx - $N] = LaplCoeff[1][2]/(dy^2)
                    Jmat[idx, idx + $N] = LaplCoeff[1][2]/(dy^2)

                    Jmat[idx + grid_size, idx + grid_size - 1] = LaplCoeff[2][1]/(dx^2)
                    Jmat[idx + grid_size, idx + grid_size + 1] = LaplCoeff[2][1]/(dx^2)
                    Jmat[idx + grid_size, idx + grid_size - $N] = LaplCoeff[2][2]/(dy^2)
                    Jmat[idx + grid_size, idx + grid_size + $N] = LaplCoeff[2][2]/(dy^2)
                end
            end

            if typeof(J) === typeof(Jmat)
                copy!(J, Jmat)
            else
                J .= Jmat
            end

            return J
        end
    end
    return eval(function_code)
end

DiffMat = [Any[sin_coeff, sin_coeff], Any[cos_coeff, cos_coeff]]
LaplCoeff = [Any[sin_coeff, sin_coeff], Any[cos_coeff, cos_coeff]]

for eqs in DiffMat
    eqs[1] = expand_derivatives(Differential(A(x, y))(eqs[1]))
    eqs[2] = expand_derivatives(Differential(B(x, y))(eqs[2]))

    eqs[1] = transform_sym_coeff(Meta.parse(string(eqs[1])))
    eqs[2] = transform_sym_coeff(Meta.parse(string(eqs[2])))
end

LaplCoeff[1][1] = Symbolics.coeff(LaplCoeff[1][1], Differential(x)(Differential(x)(A(x, y))))
LaplCoeff[1][2] = Symbolics.coeff(LaplCoeff[1][2], Differential(y)(Differential(y)(A(x, y))))
LaplCoeff[2][1] = Symbolics.coeff(LaplCoeff[2][1], Differential(x)(Differential(x)(B(x, y))))
LaplCoeff[2][2] = Symbolics.coeff(LaplCoeff[2][2], Differential(y)(Differential(y)(B(x, y))))


sin_ex = transform_sym(Meta.parse(string(sin_coeff)))
cos_ex = transform_sym(Meta.parse(string(cos_coeff)))



println(sin_coeff)

println(DiffMat[1][1])
println(DiffMat[1][2])
println(DiffMat[2][1])
println(DiffMat[2][2])

println(LaplCoeff[1][1])


nonlinear_residual = create_residual_function(N, sin_ex, cos_ex)
initial_guess = ones(2*N*N)
par = [stepx, stepy]
jacobian = create_jacobian_function(N, DiffMat, LaplCoeff)

# nonlinear_function = NonlinearFunction(nonlinear_residual, jac=jacobian)
nonlinear_function = NonlinearFunction(nonlinear_residual)

prob = NonlinearProblem(nonlinear_function, initial_guess, par)
println("Starting nonlinear solver...")
@time sol = NonlinearSolve.solve(prob, NewtonRaphson(), reltol = 1e-5, abstol = 1e-5)
println("Nonlinear solver finished!")

#print(sol.u)


A_sol = reshape(sol.u[1:N*N], N, N)
B_sol = reshape(sol.u[N*N+1:2*N*N], N, N)


xgrid = range(xleft, xright, length=N)
ygrid = range(yleft, yright, length=N)


tgrid = 0.0:0.1:30.0
total_frames = length(tgrid)


anim = @animate for t in tgrid
    u_t = A_sol .* sin(omega*t) .+ B_sol .* cos(omega*t)
    heatmap(xgrid, ygrid, u_t',
            color=:magma,
            xlabel="x", ylabel="y",
            title="u(x,y,t) at t=$t s",
            clims=(-7.5, 7.5),
            aspect_ratio=1)
end

gif(anim, "wave_jac.gif", fps=25)

