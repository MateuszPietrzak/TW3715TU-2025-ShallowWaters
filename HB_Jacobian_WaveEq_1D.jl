import ModelingToolkit as Model
using Symbolics
using DomainSets
import ApproxFun as AF
using NonlinearSolve
import DifferentialEquations as DE
using MacroTools: @capture, postwalk, prewalk
using SparseArrays
using Plots



gamma = 1.0;
omega = 0.5;
gamma3 = 1.0;
g0::Float64 = 9.80665; # m / s^2
height = 5.0; # m


xleft::Float64 = 0.0;
xright::Float64 = 1.0;
Nt = 5
N = Nx = 75;
harmonics = 3; # number of harmonics
order = 2;
stepx = (xright-xleft)/Nx;
u0 = 250 * ones((Nx+1) * harmonics * 2);
# N = (Nx+1) * (Ny+1);


# Define symbolics
Model.@parameters x, y, t;

const Dy = Model.Differential(y)
const Dx = Model.Differential(x);
const Dt = Model.Differential(t);


function build_problem(x, t, omega, harmonics, xleft, xright, gamma, gamma3)
    vars, var_exprs, (u,) = create_ansatz((x,), t, omega, harmonics);
    F = 250 * exp(-40*(x^2)) * sin(omega*t);
    pde::Symbolics.Num = Dt(Dt(u)) - 0.25*(Dx(Dx(u))) + gamma*Dt(u) + gamma3*Dt(u)*Dt(u)*Dt(u) - F;
    return pde, var_exprs, vars;
end

function simplify_problem(pde, t, omega, harmonics, Nx, vars)
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
        if @capture(instr, Differential(x)(Differential(x)(s_(x))))
            return :(($s[i+1]-2* $s[i]+$s[i-1])/dx^2)
        elseif @capture(instr, Differential(x)(s_(x)))
            return :(($s[i+1] - $s[i-1]) / (2*dx))
        elseif @capture(instr, s_(x))
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[i, $(number)])
        elseif @capture(instr, s_[i_])
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[$(i), $(number)])
        elseif @capture(instr, x)
            return :(i * dx)
        end
        return instr
    end
end

function transform_sym_coeff(ex)
    if !isa(ex, Expr) return ex end
    return prewalk(ex) do instr
        if @capture(instr, s_(x))
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[i, $(number)])
        elseif @capture(instr, x)
            return :(i * dx)
        end
        return instr
    end
end

function create_residual_function(N, equationsExpr, harmonics)
    equationsExprMapped::Vector{Expr} = []

    println(length(equationsExpr))
    println(harmonics)

    for H in 1:harmonics
        push!(equationsExprMapped, :(F_view[i,1,$(H)] = $(equationsExpr[2*H - 1]))) # Related to equationsExpr[2*H-1]
        push!(equationsExprMapped, :(F_view[i,2,$(H)] = $(equationsExpr[2*H]))) # Related to equationsExpr[2*H]

    end

    function_code = quote
        function residual!(F, U, p)
            dx = p[1]
            grid_size = $N
            harmonicsNum = $harmonics


            A_array = reshape(@view(U[1:harmonicsNum*grid_size]), $N, harmonicsNum)
            B_array = reshape(@view(U[harmonicsNum*grid_size+1:2*harmonicsNum*grid_size]), $N, harmonicsNum)


            F_view = reshape(@view(F[1:end]), $N, 2, harmonicsNum)


            # Inner points:
            for i in 2:$(N-1)
                $(equationsExprMapped...)
            end

            # BCs (Dirichlet for now):
            for H in 1:harmonicsNum
                F_view[1,1,H] = A_array[1,H]; F_view[end,1,H] = A_array[end,H]
                F_view[1,2,H] = B_array[1,H]; F_view[end,2,H] = B_array[end,H]
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
            println("___________________________")
            println(temp)
            println("___________________________")
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
            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size] = $(DiffMat[2*HEq*harmonics+HHar]) - 2*$(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))

            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size - 1] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))

            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size + 1] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))
        end
    end

    function_code = quote
        function jacobian!(J, U, p)
            dx = p[1]
            grid_size = $N
            harmonicsNum = $(harmonics)

            A_array = reshape(@view(U[1:harmonicsNum*grid_size]), $N, harmonicsNum)
            B_array = reshape(@view(U[harmonicsNum*grid_size+1:2*harmonicsNum*grid_size]), $N, harmonicsNum)

            Jmat = spzeros(2*grid_size*harmonicsNum, 2*grid_size*harmonicsNum)

            #Boundary conditions (easier to write over all diagonal points, then overwrite later):
            for k in 1:2*grid_size*harmonicsNum
                Jmat[k,k] = 1
            end

            #Inner points:
            for i in 2:$(N-1)
                idx = i
                $(equationsExprMapped...)
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


pde, var_exprs, vars = build_problem(x, t, omega, harmonics, xleft, xright, gamma, gamma3)
equations = simplify_problem(pde, t, omega, harmonics, Nx, vars)
equationsExpr = create_discretized(equations, harmonics)
println(equationsExpr[1])
println("_____________________________")
println(equationsExpr[2])
println("_____________________________")
DiffMat, LaplCoeff = create_jac_blocks(equations, var_exprs, harmonics)



println(DiffMat[1])
println("_____________________________")
println(DiffMat[2])
println("_____________________________")
println(DiffMat[3])
println("_____________________________")
println(DiffMat[4])
println("_____________________________")

println(LaplCoeff[3])





nonlinear_residual = create_residual_function(N, equationsExpr, harmonics)
initial_guess = zeros(2*N*harmonics)
par = [stepx]
jacobian = create_jacobian_function(N, DiffMat, LaplCoeff, harmonics)

nonlinear_function = NonlinearFunction(nonlinear_residual, jac=jacobian)
# nonlinear_function = NonlinearFunction(nonlinear_residual)

prob = NonlinearProblem(nonlinear_function, initial_guess, par)
println("Starting nonlinear solver...")
@time sol = NonlinearSolve.solve(prob, NewtonRaphson(), reltol = 1e-5, abstol = 1e-5, maxiters=10)
println("Nonlinear solver finished!")

#print(sol.u)


A_sol = reshape(sol.u[1:N], N)
B_sol = reshape(sol.u[N+1:2*N], N)


xgrid = range(xleft, xright, length=N)


tgrid = 0.0:0.1:30.0
total_frames = length(tgrid)


anim = @animate for t in tgrid
    u_t = A_sol .* sin(omega*t) .+ B_sol .* cos(omega*t)

    plot(xgrid, u_t,
         xlabel = "x",
         ylabel = "u(x,t)",
         title = "u(x,t) at t = $(round(t, digits=2)) s",
         ylim = (-7.5, 7.5),
         legend = false,
         linewidth = 2)
end


gif(anim, "wave_1D.gif", fps=25)

