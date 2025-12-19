using MethodOfLines
using NonlinearSolve
import ModelingToolkit as Model
import SymPy as sp
import Symbolics as Symb
using DomainSets
import ApproxFun as AF
import DifferentialEquations as DE
using Symbolics

function create_ansatz(coords::Tuple, t::Symbolics.Num, omega, harmonics::Int)
    var_names = [Symbol(Char('A' + i - 1)) for i in 1:(2*harmonics)]
    
    var_exprs = Symbolics.Num[]
    
    u::Num = Num(0)
    j = 1
    
    for (i, name) in enumerate(var_names)
        v::Symbolics.CallWithMetadata{SymbolicUtils.FnType{Tuple, Real}, Base.ImmutableDict{DataType, Any}} = first(@variables $name(..))
        expr::Num = v(coords...)

        if isodd(i)
            u += expr * sin(j * omega * t)
        else
            u += expr * cos(j * omega * t)
            j += 1
        end
        
        push!(var_exprs, expr)
    end
    
    return var_names, var_exprs, u  # Return names instead of callables
end

function create_bcs(var_names::Vector{Symbol}, domains, indep_vars, bc)
    ndims = length(domains)
    bcs = Equation[]
    
    for name in var_names
        # Recreate the callable from the name
        v = first(@variables $name(..))
        
        if ndims == 1
            x = indep_vars[1]
            xleft, xright = domains[1]
            push!(bcs, v(xleft) ~ bc)
            push!(bcs, v(xright) ~ bc)
        elseif ndims == 2
            x, y = indep_vars
            xleft, xright = domains[1]
            yleft, yright = domains[2]
            push!(bcs, v(xleft, y) ~ bc)
            push!(bcs, v(xright, y) ~ bc)
            push!(bcs, v(x, yleft) ~ bc)
            push!(bcs, v(x, yright) ~ bc)
        end
    end
    
    return bcs
end



function expand_trig(eqn, t, omega, verbose=false)
    y_exp = Symb.expand(Model.expand_derivatives(eqn))
    symbolics_list = Symb.arguments(y_exp, +)
    
    println("Number of terms: ", length(symbolics_list))
    
    t_sympy = sp.symbols("t")
    finished_terms = Num[]
    
    for (i, term) in enumerate(symbolics_list)
            term_sympy = Symb.symbolics_to_sympy(term)
            current_term = term_sympy.as_independent(t_sympy)[2]
            fs = sp.sympy.fourier_series(current_term, (t_sympy, -sp.PI, sp.PI))
            finished_sympy_term = fs.truncate() / current_term 
            finished_symb_term = Symb.sympy_to_symbolics(finished_sympy_term, [t]) * symbolics_list[i]
            push!(finished_terms, finished_symb_term)
        println("Finished term $i")
    end
    
    finished_terms = [Symb.simplify(term) for term in finished_terms]
    return sum(finished_terms)
end

function expand_trig_jl(eqn, t, omega)
    y_exp = Symb.expand(Model.expand_derivatives(eqn))
    symbolics_list = Symb.arguments(y_exp, +)
    contains_var(expr, var) = any(v -> isequal(v, var), Symbolics.get_variables(expr))
    finished_terms = Num[]
    
    for (i, term) in enumerate(symbolics_list)
        trig_terms = Num[]      # Typed
        spatial_terms = Num[]   # Typed
        
        for mul_term in Symb.arguments(term, *)
            mul_term_num = Num(mul_term)  # Ensure Num type
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
        coeffs = AF.coefficients(F)::Vector{Float64}  # Type assert
        
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

function make_residual(expanded, harmonics, omega, t)
    eqs = Equation[]
    for i in 1:harmonics
        sin_coef::Num = Symb.coeff(expanded, sin(i*omega*t))
        cos_coef::Num = Symb.coeff(expanded, cos(i*omega*t))
        push!(eqs, sin_coef ~ 0)
        push!(eqs, cos_coef ~ 0)
    end
    return eqs
end


using ModelingToolkit

# Harmonic Balance solver using MethodOfLines -> very slow
function solve_harmonicbalance(eqs, harmonics, bcs, domains, steps, vars, vars_symb, x0)
    ndims = length(vars)
    
    domain_specs = [vars[i] ∈ Interval(domains[i]...) for i in 1:ndims]
    
    Model.@named pdesys = Model.PDESystem(eqs, bcs, domain_specs, collect(vars), vars_symb)
    
    disc_dict = [vars[i] => steps[i] for i in 1:ndims]
    @time discretization = MOLFiniteDifference(disc_dict, nothing, approx_order=2)
    
    @time sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    sys::Model.System = Model.complete(sys)

    unknowns::Vector{SymbolicUtils.BasicSymbolic{Real}} = Model.unknowns(sys)

    
    u0_map = Dict(unknowns .=> x0)
    
    @time prob = NonlinearProblem(sys, u0_map; discretization.kwargs...)
    
    @time sol = solve(prob_sparse, NewtonRaphson(; linsolve=KLUFactorization()))
        
    # @time sol = NonlinearSolve.solve(prob, NewtonRaphson(), reltol=1e-5, abstol=1e-5)

    # Type the output
    solution_coeffs = Matrix{Float64}[sol[vars_symb[i]] for i in 1:(2*harmonics)]
    
    return solution_coeffs
end




# Implementing with the custom stencil
function solve_harmonicbalance_DiffEq(eqs, harmonics::Int, bcs, domains, steps, vars, vars_symb, x0)
    ndims = length(vars)
    
    domain_specs = [vars[i] ∈ Interval(domains[i]...) for i in 1:ndims]
    
    @named pdesys = PDESystem(eqs, bcs, domain_specs, collect(vars), vars_symb)
    
    disc_dict = Dict(vars[i] => steps[i] for i in 1:ndims)
    discretization = MOLFiniteDifference(disc_dict, nothing; approx_order=2)
    
    prob = discretize(pdesys, discretization)  # Use discretize for NonlinearProblem
    
    prob_with_u0 = remake(prob; u0=x0)
    
    sol = solve(prob_with_u0, NewtonRaphson(); reltol=1e-5, abstol=1e-5)
    
    solution_coeffs = [Vector{Float64}(sol[vars_symb[i]]) for i in 1:(2*harmonics)]
    
    return solution_coeffs
end

# using Plots
# anim = @animate for t in 1:0.1:10
#     if t % 100 == 0
#         println(t)
#     end

#     u_new = solution_coeffs[1]*0.0
#     j = 1
#     for i in 1:(2*harmonics)
#         if isodd(i)
#             u_new .+= solution_coeffs[i] .* sin(j * omega * t)
#         else
#             u_new .+= solution_coeffs[i] .* cos(j * omega * t)
#             j += 1
#         end
#     end
    

#     heatmap(u_new, clims=(-0.1, 0.1))
#     title!("(Solvetime = 296s) Wave Equation with H = $(harmonics), \$\\gamma\$ = $(gamma), \$\\gamma_3\$ = $(gamma3), \$\\omega\$ = $omega", titlefontsize=10)

# end

# gif(anim, "HB_WE_2D.gif", fps=30)
