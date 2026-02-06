using ModelingToolkit

# OLD code, keep for consistency and reference
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

# OLD code, keep for consistency and reference
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