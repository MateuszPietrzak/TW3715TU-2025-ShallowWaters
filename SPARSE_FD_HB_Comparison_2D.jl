import ModelingToolkit as Model
using Symbolics
using DomainSets
import ApproxFun as AF
using NonlinearSolve
import DifferentialEquations as DE
using MacroTools: @capture, postwalk, prewalk
using SparseArrays
using Plots



gamma = 0.0;
omega = 1.0;
gamma3 = 0.0;
g0::Float64 = 9.80665; # m / s^2
height = 5.0; # m


xleft::Float64 = 0.0;
xright::Float64 = 10.0;
yleft = 0.0;
yright = 10.0;
Nt = 5
N = Nx = Ny = 150;
harmonics = 1; # number of harmonics
order = 1;
stepx = (xright-xleft)/Nx;
stepy = (yright - yleft)/Ny;
u0 = 250 * ones((Nx) * (Ny) * harmonics * 2);
# N = (Nx+1) * (Ny+1);


# Define symbolics
Model.@parameters x, y, t;

const Dy = Model.Differential(y)
const Dx = Model.Differential(x);
const Dt = Model.Differential(t);


function build_problem(x, y, t, omega, harmonics, xleft, xright, yleft, yright, gamma, gamma3)
    vars, var_exprs, (u,) = create_ansatz((x, y), t, omega, harmonics);
    F = 250 * exp(-40*((x/10)^2)) * sin(omega*t);
    pde::Symbolics.Num = Dt(Dt(u)) - 0.25*(Dx(Dx(u)) + Dy(Dy(u))) + gamma*Dt(u) + gamma3*Dt(u)*Dt(u)*Dt(u) - F;
    return pde, var_exprs, vars;
end

function simplify_problem(pde, t, omega, harmonics, Nx, Ny, vars)
    expanded = expand_trig_jl(pde, t, omega)
    eqns = make_equations(expanded, harmonics, omega, t)
    return eqns
end


function create_ansatz(coords::Tuple, t::Symbolics.Num, omega, harmonics::Int, n_fields::Int=1)
    var_names = Symbol[]
    var_exprs = Symbolics.Num[]
    fields = Symbolics.Num[]

    letter_idx = 1

    for field_idx in 1:n_fields
        u = Num(0)
        j = 1

        for i in 1:(2*harmonics)
            if isodd(i)
                name = Symbol("A", div(i + 1, 2))
            else
                name = Symbol("B", div(i, 2))
            end
            letter_idx += 1

            v = first(@variables $name(..))
            expr = v(coords...)

            if isodd(i)
                u += expr * sin(j * omega * t)
            else
                u += expr * cos(j * omega * t)
                j += 1
            end

            push!(var_names, name)
            push!(var_exprs, expr)
        end

        push!(fields, u)
    end

    return var_names, var_exprs, fields
end


function expand_trig_jl(eqn, t, omega)
    y_exp = expand(Model.expand_derivatives(eqn))
    symbolics_list = arguments(y_exp, +)
    contains_var(expr, var) = any(v -> isequal(v, var), Symbolics.get_variables(expr))
    finished_terms = Num[]

    for (i, term) in enumerate(symbolics_list)
        trig_terms = Num[]
        spatial_terms = Num[]

        for mul_term in arguments(term, *)
            mul_term_num = Num(mul_term)
            if contains_var(mul_term_num, t)
                push!(trig_terms, mul_term_num)
            else
                push!(spatial_terms, mul_term_num)
            end
        end

        spatial::Num = isempty(spatial_terms) ? Num(1) : prod(spatial_terms)

        if length(trig_terms) == 1
            unwrapped = Symbolics.unwrap(trig_terms[1])
            if SymbolicUtils.operation(unwrapped) !== ^
                push!(finished_terms, Num(term))
            continue
            end
        end

        trig::Num = isempty(trig_terms) ? Num(1) : prod(trig_terms)

        trig_func = Symbolics.build_function(trig, t, expression=Val{false})
        # trig_func = x -> Symbolics.value(Symbolics.substitute(trig, t => x))
        period = 2π / omega
        F = AF.Fun(trig_func, AF.Fourier(-period/2 .. period/2))
        coeffs = AF.coefficients(F)::Vector{Float64}

        ωt = omega * t
        expanded_trig::Num = Num(0)

        for (j, c) in enumerate(coeffs)
            abs(c) < 1e-10 && continue

            if j == 1
                expanded_trig += c
                elseif iseven(j)
                n = j ÷ 2
                expanded_trig += c * cos(n * ωt)
            else
                n = (j + 1) ÷ 2
                expanded_trig += c * sin(n * ωt)
            end
        end

        push!(finished_terms, spatial * -expanded_trig)
    end

    return sum(finished_terms)
end

function make_equations(expanded, harmonics, omega, t)
    eqs = []
    for i in 1:harmonics
        sin_coef = Symbolics.coeff(expanded, sin(i*omega*t))
        cos_coef = Symbolics.coeff(expanded, cos(i*omega*t))
        push!(eqs, sin_coef)
        push!(eqs, cos_coef)
    end
    return eqs
end

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
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[i, j, $(number)])
            elseif @capture(instr, s_[i_, j_])
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[$(i), $(j), $(number)])
            elseif @capture(instr, x)
            return :(i * dx)
            elseif @capture(instr, y)
            return :(j * dy)
        end
        return instr
    end
end

function transform_sym_coeff(ex)
    if !isa(ex, Expr) return ex end
    return prewalk(ex) do instr
        if @capture(instr, s_(x, y))
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[i, j, $(number)])
            elseif @capture(instr, x)
            return :(i * dx)
            elseif @capture(instr, y)
            return :(j * dy)
        end
        return instr
    end
end

function create_residual_function(N, equationsExpr, harmonics)
    equationsExprMapped::Vector{Expr} = []

    println(length(equationsExpr))
    println(harmonics)

    for H in 1:harmonics
        push!(equationsExprMapped, :(F_view[i, j,1,$(H)] = $(equationsExpr[2*H - 1]))) # Related to equationsExpr[2*H-1]
        push!(equationsExprMapped, :(F_view[i, j,2,$(H)] = $(equationsExpr[2*H]))) # Related to equationsExpr[2*H]

    end

    function_code = quote
        function residual!(F, U, p)
            dx, dy = p
            grid_size = $N * $N
            harmonicsNum = $harmonics


            A_array = reshape(@view(U[1:harmonicsNum*grid_size]), $N, $N, harmonicsNum)
            B_array = reshape(@view(U[harmonicsNum*grid_size+1:2*harmonicsNum*grid_size]), $N, $N, harmonicsNum)


            F_view = reshape(@view(F[1:end]), $N, $N, 2, harmonicsNum)


            # Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    $(equationsExprMapped...)
                end
            end

            # BCs (Dirichlet for now):
            for H in 1:harmonicsNum
                F_view[1,:,1,H] .= A_array[1,:,H]; F_view[end,:,1,H] .= A_array[end,:,H]
                F_view[:,1,1,H] .= A_array[:,1,H]; F_view[:,end,1,H] .= A_array[:,end,H]
                F_view[1,:,2,H] .= B_array[1,:,H]; F_view[end,:,2,H] .= B_array[end,:,H]
                F_view[:,1,2,H] .= B_array[:,1,H]; F_view[:,end,2,H] .= B_array[:,end,H]
            end

            return F
        end
    end
    res = eval(function_code)
    return res
end

function always_expr(s::String)
    e = Meta.parse(s)
    return isa(e, Expr) ? e : :($e)
end

function create_jac_blocks(equations, amplitudes, harmonicsNum)
    DiffMat = Vector{Union{Expr, Float64, Int64}}(undef, (2*harmonicsNum)^2)
    LaplCoeff = Vector{Union{Expr, Float64, Int64}}(undef, ((2*harmonicsNum)^2)*2)
    for i in 1:2*harmonicsNum
        for j in 1:2*harmonicsNum
            index = (i-1)*2*harmonicsNum + j
            temp = Meta.parse(string(expand_derivatives(Differential(amplitudes[j])(equations[i]))))
            temp2 = transform_sym_coeff(temp)
            DiffMat[index] = temp2

            temp = always_expr(string(Symbolics.coeff(equations[i], Differential(x)(Differential(x)(amplitudes[j])))))
            LaplCoeff[index*2 - 1] = transform_sym_coeff(temp)
            temp = always_expr(string(Symbolics.coeff(equations[i], Differential(y)(Differential(y)(amplitudes[j])))))
            LaplCoeff[index*2] = transform_sym_coeff(temp)
        end
    end
    return DiffMat, LaplCoeff
end


function create_discretized(equations, harmonicsNum)
    equationsExpr = Vector{Expr}(undef, 2*harmonicsNum)
    for i in 1:2*harmonicsNum
        equationsExpr[i] = transform_sym(always_expr(string(equations[i])))
    end
    return equationsExpr
end


function create_jacobian_function(N, DiffMat, LaplCoeff, harmonics)
    equationsExprMapped::Vector{Expr} = []

    for HEq in 0:2*harmonics-1
        for HHar in 1:2*harmonics
            push!(equationsExprMapped, :(J[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size] = $(DiffMat[2*HEq*harmonics+HHar]) - 2*$(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2) - 2*$(LaplCoeff[2*(2*HEq*harmonics+HHar)])/(dy^2)))

            push!(equationsExprMapped, :(J[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size - 1] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))

            push!(equationsExprMapped, :(J[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size + 1] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))

            push!(equationsExprMapped, :(J[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size - $(N)] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)])/(dy^2)))

            push!(equationsExprMapped, :(J[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size + $(N)] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)])/(dy^2)))
        end
    end

    function_code = quote
        function jacobian!(J, U, p)
            dx, dy = p
            grid_size = $N * $N
            harmonicsNum = $(harmonics)

            A_array = reshape(@view(U[1:harmonicsNum*grid_size]), $N, $N, harmonicsNum)
            B_array = reshape(@view(U[harmonicsNum*grid_size+1:2*harmonicsNum*grid_size]), $N, $N, harmonicsNum)

            fill!(J, 0.0)

            #Boundary conditions (easier to write over all diagonal points, then overwrite later):
            for k in 1:2*grid_size*harmonicsNum
                J[k,k] = 1
            end

            #Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    idx = (j-1)*$N + i
                    $(equationsExprMapped...)
                end
            end

            return J
        end
    end
    return eval(function_code)
end


pde, var_exprs, vars = build_problem(x, y, t, omega, harmonics, xleft, xright, yleft, yright, gamma, gamma3)
equations = simplify_problem(pde, t, omega, harmonics, Nx, Ny, vars)
equationsExpr = create_discretized(equations, harmonics)
DiffMat, LaplCoeff = create_jac_blocks(equations, var_exprs, harmonics)


nonlinear_residual = create_residual_function(N, equationsExpr, harmonics)
initial_guess = u0
par = [stepx, stepy]
JacSize = 2 * N * N * harmonics
jac_prototype = spzeros(JacSize, JacSize)
jacobian = create_jacobian_function(N, DiffMat, LaplCoeff, harmonics)

nonlinear_function = NonlinearFunction(nonlinear_residual, jac=jacobian, jac_prototype = jac_prototype)
# nonlinear_function = NonlinearFunction(nonlinear_residual)

prob = NonlinearProblem(nonlinear_function, initial_guess, par)
println("Starting nonlinear solver...")
@time sol = NonlinearSolve.solve(prob, NewtonRaphson(), reltol = 1e-5, abstol = 1e-5, maxiters=10)
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
            clims=(-900, 900),
            aspect_ratio=1)
end

gif(anim, "wave_jac.gif", fps=25)

gamma_3 = 0.0
@variables u(..)
c = 0.5
y_eq = c*c*(Dx(Dx(u(x, y))) + Dy(Dy(u(x, y)))) - gamma*Dt(u(x, y)) - gamma_3*Dt(u(x, y))*Dt(u(x, y))*Dt(u(x, y))
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
                    ddu_array[i, j] = ($y_expr) - 250 * exp(-40*((i * dx / 10))^2) * sin(t)
                end
            end

            return ddu
        end
    end
    return eval(function_code)
end

y_expr = transform_sym(Meta.parse(string(y_eq)))

ODEfunc = create_ODE_function(N, y_expr)
u0_FD = zeros(N*N)
du0 = zeros(N*N)
tspan = (0.0, 200.0)
par = [stepx, stepy]
prob = SecondOrderODEProblem(ODEfunc, du0, u0_FD, tspan, par)

@time sol = solve(prob, DE.Tsit5(), saveat=0.01, progress=true, progress_steps=200)

tgrid = sol.t


xgrid = range(xleft, xright, length=N)
ygrid = range(yleft, yright, length=N)


u_data = [reshape(sol.u[k].x[1], N, N) for k in eachindex(sol.t)]
#
# anim = @animate for (i, t) in enumerate(tgrid)
#     heatmap(xgrid, ygrid, u_data[i],
#             color=:magma,
#             xlabel="x", ylabel="y",
#             title="u(x,y,t) at t=$t s",
#             clims=(-900, 900),
#             aspect_ratio=1)
# end
#
# gif(anim, "FD_sol.gif", fps=25)

fs_time = fs = 1/0.01
t_steady0 = Int(round(length(u_data)*0.8))
target = 1 * omega / (2 * pi)
println(length(u_data))
function generateComplexAmplitudeFDSpectrumMatrix(sol, target)
r1 = zeros(N, N)
rows = 0
println(rows)
for x_idx in 1:1:N
    for y_idx in 1:1:N
        u_at_point = [sol[t][x_idx, y_idx] for t in 1:length(sol)]
            F = fftshift(fft(u_at_point))
            freqs = fftshift(fftfreq(length(u_at_point), fs_time))
            (_, idx) = findmin(abs.(freqs .- target))
            r1[x_idx, y_idx] = abs(F[idx]) / (length(sol) - t_steady0)
        end
    end
    r1
end

using FFTW
fd_complex_spectrum = generateComplexAmplitudeFDSpectrumMatrix(u_data, target)
heatmap(fd_complex_spectrum)
savefig("FD_Complex_Spectrum.png")

coefficientsHB = []
push!(coefficientsHB, A_sol)
push!(coefficientsHB, B_sol)
println(size(coefficientsHB[1]))

function generateComplexAmplitudeHBSpectrumMatrix(h)
    r1 = zeros(N, N)
    rows = 0
    for x_idx in 1:1:N
        for y_idx in 1:1:N
            #f = h * omega / 2π
            r1[x_idx, y_idx] = abs(0.5 * coefficientsHB[2 * h][x_idx, y_idx] - im * 0.5 * coefficientsHB[2 * h - 1][x_idx, y_idx])
        end
    end
    r1
end
hb_complex_spectrum = generateComplexAmplitudeHBSpectrumMatrix(1)
heatmap(hb_complex_spectrum)
savefig("HB_Complex_Spectrum.png")
