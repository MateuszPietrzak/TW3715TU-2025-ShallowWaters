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
N = 500
stepx = (xright-xleft)/(N-1)
stepy = (yright-yleft)/(N-1)

#Definition of constants (placeholders):
gamma::Float64 = 0.1
gamma_3::Float64 = 0.1
c::Float64 = 0.5
omega::Float64 = 1

@parameters x, y, t
@variables u(..)

Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)
#Full PDE: Dt(Dt(u(x, y))) - c*c*(Dx(Dx(u(x, y))) + Dy(Dy(u(x, y)))) + gamma*Dt(u(x, y)) + gamma_3*Dt(u(x, y))*Dt(u(x, y))*Dt(u(x, y))
#Reformat as ddu = y_eq
y_eq = c*c*(Dx(Dx(u(x, y))) + Dy(Dy(u(x, y)))) - gamma*Dt(u(x, y)) - gamma_3*Dt(u(x, y))*Dt(u(x, y))*Dt(u(x, y))
# y_eq = expand_derivatives(y_eq)

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
            elseif @capture(instr, Differential(t)(s_(x, y)))
            return :($(Symbol("d$(s)_array"))[i, j])
            elseif @capture(instr, Differential(x, 2)(s_(x, y)))
            return :(($s[i+1, j]-2* $s[i, j]+$s[i-1, j])/dx^2)
            elseif @capture(instr, Differential(y, 2)(s_(x, y)))
            return :(($s[i, j+1]-2* $s[i, j]+$s[i, j-1])/dy^2)
            elseif @capture(instr, Differential(x, 1)(s_(x, y)))
            return :(($s[i+1, j] - $s[i-1, j]) / (2*dx))
            elseif @capture(instr, Differential(y, 1)(s_(x, y)))
            return :(($s[i, j+1] - $s[i, j-1]) / (2*dy))
            elseif @capture(instr, Differential(t, 1)(s_(x, y)))
            return :($(Symbol("d$(s)_array"))[i, j])
            elseif @capture(instr, s_(x, y))
            return :($(Symbol("$(s)_array"))[i, j])
            elseif @capture(instr, s_[i_, j_])
            return :($(Symbol("$(s)_array"))[$(i), $(j)])
        end
        return instr
    end
end
println(y_eq)
println("_____________________________________________")
println(transform_sym(Meta.parse(string(y_eq))))

function create_ODE_function(N, y_expr)
    function_code = quote
        function secODE!(ddu, du, u, p, t)
            dx, dy = p
            grid_size = $N * $N

            u_array = reshape(@view(u[1:grid_size]), $N, $N)
            du_array = reshape(@view(du[1:grid_size]), $N, $N)
            ddu_array = reshape(@view(ddu[1:grid_size]), $N, $N)

            # BCs (Dirichlet for now):
            ddu_array[1,:] .= 0; ddu_array[end,:] .= 0
            ddu_array[:,1] .= 0; ddu_array[:,end] .= 0

            # Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    ddu_array[i, j] = ($y_expr) - 10000 * c^2 * exp(-400*(i * dx)^2) * sin(t)
                end
            end

            return ddu
        end
    end
    return eval(function_code)
end

y_expr = transform_sym(Meta.parse(string(y_eq)))

ODEfunc = create_ODE_function(N, y_expr)
u0 = zeros(N*N)
du0 = zeros(N*N)
tspan = (0.0, 60.0)
par = [stepx, stepy]
prob = SecondOrderODEProblem(ODEfunc, du0, u0, tspan, par)

@time sol = solve(prob, DPRKN6(), saveat=0.1)

tgrid = sol.t


xgrid = range(xleft, xright, length=N)
ygrid = range(yleft, yright, length=N)


u_data = [reshape(sol.u[k].x[1], N, N) for k in eachindex(sol.t)]

anim = @animate for (i, t) in enumerate(tgrid)
    heatmap(xgrid, ygrid, u_data[i],
            color=:magma,
            xlabel="x", ylabel="y",
            title="u(x,y,t) at t=$t s",
            clims=(-7.5, 7.5),
            aspect_ratio=1)
end

gif(anim, "FD_sol.gif", fps=25)
