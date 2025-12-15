using MethodOfLines
using NonlinearSolve
import ModelingToolkit as Model
import SymPy as sp
import Symbolics as Symb
using DomainSets
import ApproxFun as AF
import DifferentialEquations as DE

function create_ansatz(x, t, omega, harmonics)
    u = 0
    j = 1
    vars = []      # Will store callable functions like A, B, C...
    var_exprs = [] # Will store expressions like A(x), B(x), C(x)...
    
    for i in 1:(2*harmonics)
        c = Char('A' + i - 1)
        v = eval(:(Model.@variables $(Symbol(c))(..)))
        
        if isodd(i)
            u += v[1](x) * sin(j * omega * t)
        else
            u += v[1](x) * cos(j * omega * t)
            j += 1
        end
        
        push!(vars, v[1])        # Store the callable function
        push!(var_exprs, v[1](x)) # Store the expression with x
    end
    
    return vars, var_exprs, u
end

function create_bcs(vars, domain, bc)
    xleft, xright = domain
    bcs = []
    for var in vars
        push!(bcs, var(xleft) ~ bc)  # Now var is callable
        push!(bcs, var(xright) ~ bc)
    end
    return bcs
end

function expand_trig(eqn, x, t, omega, verbose=false)
    """
    The idea for this new implementation is to use the form of the PDE with the ansatz substituted in 
    and extract each term (separate wrt addition/subtraction)
    
    Each term is then converted from the Symbolics.jl form to the SymPy form and the part dependent 
    on t is extracted for the Fourier series. The expanded Fourier Series term is divided by the 
    SymPy non-simplified term. Convert back to Symbolics.jl this form and by multiplying it with 
    the original "term" the trigonometric term to the power (e.g. cos(x)^3) is cancelled out.
    """
    y_exp = Symb.expand(Model.expand_derivatives(eqn));
    symbolics_list = Symb.arguments(y_exp, +);
    
    t_sympy = sp.symbols("t")
    x_sympy = sp.symbols("x")
    finished_terms = []  # Initialize empty list
    
    for (i, term) in enumerate(symbolics_list)
        # Transform term by term into sympy form
        term_sympy = Symb.symbolics_to_sympy(term)
        current_term = term_sympy.as_independent(t_sympy)[2] # extract term sinusoidal term dependent on t
        
        fs = sp.sympy.fourier_series(current_term, (t_sympy, -sp.PI, sp.PI)) # simplify using Sympy's Fourier Series
        finished_sympy_term = fs.truncate() / current_term 
        finished_symb_term = Symb.sympy_to_symbolics(finished_sympy_term, [t])*symbolics_list[i]
        push!(finished_terms, finished_symb_term)
    
        if i == 2 && verbose
            println("We start with the following Symbolics.jl term ", symbolics_list[i])
            println("With the sympy fourier series convert ", current_term,  " to ", fs.truncate())
            println("Now write this term as ", finished_sympy_term)
            println("and multiply it with the term from the symbolics.jl form, in such a way to simplify the cubic cosine")
            println(symbolics_list[i], "*",     finished_sympy_term)
        end
    end
    
    return Symb.simplify(sum(finished_terms))
end

function make_residual(expanded, harmonics, omega, t)
    eqs = []

    for i in 1:harmonics
        sin_coef = Symb.coeff(expanded, sin(i*omega*t));
        cos_coef = Symb.coeff(expanded, cos(i*omega*t));
        push!(eqs, sin_coef ~ 0)
        push!(eqs, cos_coef ~ 0)
    end
    return eqs
end



function solve_harmonicbalance(eqs, harmonics, bcs, domains, stepx, vars, vars_symb, x0)
    xleft, xright = domains
    domains = [x ∈ Interval(xleft, xright)];

    # Create the PDESystem
    Model.@named pdesys = Model.PDESystem(eqs, bcs, domains, [vars], vars_symb);

    # Discretization
    discretization = MOLFiniteDifference([x => stepx], nothing, approx_order=2);

    # Convert to NonlinearProblem
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization);
    
    sys = Model.complete(sys)
    
    prob = NonlinearProblem(sys, ones(length(Model.get_eqs(sys))); discretization.kwargs...);
    
    # Remake and solve
    prob_new = remake(prob, u0=x0)
    sol = NonlinearSolve.solve(prob_new, NewtonRaphson(), reltol=1e-5, abstol=1e-5)

    solution_coeffs = []
    for i in 1:(2*harmonics)
        # Extract solution    
        push!(solution_coeffs, sol[vars_symb[i]]);
    end

    return solution_coeffs
end


