import ModelingToolkit as Model
using Symbolics
using DomainSets
import ApproxFun as AF
using NonlinearSolve
import DifferentialEquations as DE
using MacroTools: @capture, postwalk, prewalk
using SparseArrays
using Plots



gamma = 0.1
omega = 1.0
gamma3 = 0.0
g::Float64 = 9.80665 # m / s^2
height = 5.0;# m


xleft::Float64 = 0.0
xright::Float64 = 10.0
yleft = 0.0
yright = 10.0
Nt = 5
N = Nx = Ny = 40
Nvar = 3
harmonics = 2; # number of harmonics
order = 1;
stepx = (xright-xleft)/Nx;
stepy = (yright - yleft)/Ny;
u0 = 250 * ones((Nx) * (Ny) * harmonics * 2 * Nvar)
# N = (Nx+1) * (Ny+1);


# Define symbolics
Model.@parameters x, y, t;

const Dy = Model.Differential(y)
const Dx = Model.Differential(x);
const Dt = Model.Differential(t);


function create_ansatz(coords::Tuple, t::Symbolics.Num, omega, harmonics::Int, n_fields::Int=1)
    var_names = Symbol[]
    var_exprs = Symbolics.Num[]
    fields = Symbolics.Num[]

    letter_idx = 1

    for field_idx in 1:n_fields
        u = Num(0)
        j = 1
        base_char = Char('G' + 2*(field_idx - 1))
        for i in 1:(2*harmonics)
            if isodd(i)
                name = Symbol(Char(base_char), div(i + 1, 2))
            else
                name = Symbol(Char(base_char + 1), div(i, 2))
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

function create_residual_function(N, equationsExpr, harmonics, Nvar)
    equationsExprMapped::Vector{Expr} = []

    println(length(equationsExpr))
    println(harmonics)

    for var in 1:Nvar
        for H in 1:harmonics
            push!(equationsExprMapped, :(F_view[i, j,1,$(H),$(var)] = $(equationsExpr[(var-1)*harmonics*2 + 2*H - 1]))) # Related to equationsExpr[2*H-1]
            push!(equationsExprMapped, :(F_view[i, j,2,$(H),$(var)] = $(equationsExpr[(var-1)*harmonics*2 + 2*H]))) # Related to equationsExpr[2*H]
        end
    end

    function_code = quote
        function residual!(F, U, p)
            dx, dy = p
            grid_size = $N * $N
            harmonicsNum = $harmonics

            G_array = reshape(@view(U[1:harmonicsNum*grid_size]), $N, $N, harmonicsNum)
            H_array = reshape(@view(U[harmonicsNum*grid_size+1:2*harmonicsNum*grid_size]), $N, $N, harmonicsNum)

            I_array = reshape(@view(U[2*harmonicsNum*grid_size+1:3*harmonicsNum*grid_size]), $N, $N, harmonicsNum)
            J_array = reshape(@view(U[3*harmonicsNum*grid_size+1:4*harmonicsNum*grid_size]), $N, $N, harmonicsNum)

            K_array = reshape(@view(U[4*harmonicsNum*grid_size+1:5*harmonicsNum*grid_size]), $N, $N, harmonicsNum)
            L_array = reshape(@view(U[5*harmonicsNum*grid_size+1:6*harmonicsNum*grid_size]), $N, $N, harmonicsNum)

            F_view = reshape(@view(F[1:end]), $N, $N, 2, harmonicsNum, $(Nvar))


            # Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    $(equationsExprMapped...)
                end
            end

            #Eta BCs
            for H in 1:harmonicsNum
                if H == 1
                    F_view[1,:,1,H,1] .= G_array[1,:,H] .- 250; F_view[end,:,1,H,1] .= G_array[end,:,H]
                else
                    F_view[1,:,1,H,1] .= G_array[1,:,H]; F_view[end,:,1,H,1] .= G_array[end,:,H]
                end
                F_view[:,1,1,H,1] .= G_array[:,1,H]; F_view[:,end,1,H,1] .= G_array[:,end,H]
                F_view[1,:,2,H,1] .= H_array[1,:,H]; F_view[end,:,2,H,1] .= H_array[end,:,H]
                F_view[:,1,2,H,1] .= H_array[:,1,H]; F_view[:,end,2,H,1] .= H_array[:,end,H]
            end

            #U BCs
            for H in 1:harmonicsNum
                F_view[1,:,1,H,2] .= F_view[2,:,1,H,2]; F_view[end,:,1,H,2] .= F_view[end-1,:,1,H,2]
                F_view[:,1,1,H,2] .= F_view[:,2,1,H,2]; F_view[:,end,1,H,2] .= F_view[:,end-1,1,H,2]
                F_view[1,:,2,H,2] .= F_view[2,:,2,H,2]; F_view[end,:,2,H,2] .= F_view[end-1,:,2,H,2]
                F_view[:,1,2,H,2] .= F_view[:,2,2,H,2]; F_view[:,end,2,H,2] .= F_view[:,end-1,2,H,2]
                #                 F_view[1,:,1,H,2] .= C_array[1,:,H]; F_view[end,:,1,H,2] .= C_array[end,:,H]
                #                 F_view[:,1,1,H,2] .= C_array[:,1,H]; F_view[:,end,1,H,2] .= C_array[:,end,H]
                #                 F_view[1,:,2,H,2] .= D_array[1,:,H]; F_view[end,:,2,H,2] .= D_array[end,:,H]
                #                 F_view[:,1,2,H,2] .= D_array[:,1,H]; F_view[:,end,2,H,2] .= D_array[:,end,H]
            end

            #V BCs
            for H in 1:harmonicsNum
                F_view[:,1,1,H,3] .= K_array[:,1,H]; F_view[:,end,1,H,3] .= K_array[:,end,H]
                F_view[1,:,1,H,3] .= F_view[2,:,1,H,3]; F_view[end,:,1,H,3] .= F_view[end-1,:,1,H,3]
                F_view[:,1,2,H,3] .= L_array[:,1,H]; F_view[:,end,2,H,3] .= L_array[:,end,H]
                F_view[1,:,2,H,3] .= F_view[2,:,2,H,3]; F_view[end,:,2,H,3] .= F_view[end-1,:,2,H,3]
                #                 F_view[1,:,1,H,3] .= E_array[1,:,H]; F_view[end,:,1,H,3] .= E_array[end,:,H]
                #                 F_view[:,1,1,H,3] .= E_array[:,1,H]; F_view[:,end,1,H,3] .= E_array[:,end,H]
                #                 F_view[1,:,2,H,3] .= F_array[1,:,H]; F_view[end,:,2,H,3] .= F_array[end,:,H]
                #                 F_view[:,1,2,H,3] .= F_array[:,1,H]; F_view[:,end,2,H,3] .= F_array[:,end,H]
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

function create_discretized(equations, harmonicsNum, Nvar)
    equationsExpr = Vector{Expr}(undef, 2*harmonicsNum*Nvar)
    for i in 1:2*harmonicsNum*Nvar
        println("Original: ", equations[i])
        equationsExpr[i] = transform_sym(always_expr(string(equations[i])))
        println("FD: ", equationsExpr[i])
    end
    return equationsExpr
end


vars, var_exprs, fields = create_ansatz((x,y), t, omega, harmonics, Nvar);
eta = fields[1];
u = fields[2];
v = fields[3];

pdes = [Dt(eta) + height * Dx(u) + height * Dy(v) - 0.5*(Dx(Dx(eta)) + Dy(Dy(eta))),
        Dt(u) + g * Dx(eta),
        Dt(v) + g * Dy(eta)];

equations = []

for pde in pdes
    expanded = expand_trig_jl(pde, t, omega)
    eqns = make_equations(expanded, harmonics, omega, t)
#     println(eqns)
    append!(equations, eqns)
end

println(equations[6])

equationsExpr = create_discretized(equations, harmonics, Nvar)


nonlinear_residual = create_residual_function(N, equationsExpr, harmonics, Nvar)
initial_guess = u0
par = [stepx, stepy]
nonlinear_function = NonlinearFunction(nonlinear_residual)

prob = NonlinearProblem(nonlinear_function, initial_guess, par)
println("Starting nonlinear solver...")
@time sol = NonlinearSolve.solve(prob, NewtonRaphson(linsolve=KrylovJL_GMRES()), reltol = 1e-5, abstol = 1e-5, maxiters=10)
println("Nonlinear solver finished!")

#print(sol.u)


A_sol = reshape(sol.u[1:N*N], N, N)
B_sol = reshape(sol.u[N*N+1:2*N*N], N, N)
A1_sol = reshape(sol.u[2*N*N+1:3*N*N], N, N)
B1_sol = reshape(sol.u[3*N*N+1:4*N*N], N, N)


xgrid = range(xleft, xright, length=N)
ygrid = range(yleft, yright, length=N)


tgrid = 0.0:0.1:30.0
total_frames = length(tgrid)


anim = @animate for t in tgrid
    u_t = A_sol .* sin(omega*t) .+ B_sol .* cos(omega*t) .+ A1_sol .* sin(2*omega*t) .+ B1_sol .* cos(2*omega*t)
    heatmap(xgrid, ygrid, u_t',
            color=:magma,
            xlabel="x", ylabel="y",
            title="u(x,y,t) at t=$t s",
            clims=(-300, 300),
            aspect_ratio=1)
end

gif(anim, "SWE_noJac.gif", fps=25)

