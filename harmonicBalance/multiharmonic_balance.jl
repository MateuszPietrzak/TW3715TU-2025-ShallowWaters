using MethodOfLines
using NonlinearSolve
import ModelingToolkit as Model
import SymPy as sp
import Symbolics as Symb
using DomainSets
import ApproxFun as AF
import DifferentialEquations as DE
using Symbolics

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
    y_exp = Symb.expand(Model.expand_derivatives(eqn))
    symbolics_list = Symb.arguments(y_exp, +)
    contains_var(expr, var) = any(v -> isequal(v, var), Symbolics.get_variables(expr))
    finished_terms = Num[]
    
    for (i, term) in enumerate(symbolics_list)
        trig_terms = Num[]
        spatial_terms = Num[]
        
        for mul_term in Symb.arguments(term, *)
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

function make_residual(expanded, harmonics, omega, t)
    eqs = Num[]
    for i in 1:harmonics
        sin_coef::Num = Symb.coeff(expanded, sin(i*omega*t))
        cos_coef::Num = Symb.coeff(expanded, cos(i*omega*t))
        push!(eqs, sin_coef)
        push!(eqs, cos_coef)
    end
    return eqs
end


# Implementing with the custom stencil
using MacroTools: prewalk, @capture
using Symbolics: Differential

function transform_sym(Nx::Int64, Ny::Int64=0)
    is_2D = Ny > 0
    
    function aux(ex)
        prewalk(ex) do tmp
            if is_2D
                # 2D case
                if @capture(tmp, Differential(x)(Differential(x)(s_(x, y))))
                    return :(($s[i+1] - 2*$s[i] + $s[i-1]) / dx^2)
                    
                elseif @capture(tmp, Differential(y)(Differential(y)(s_(x, y))))
                    return :(($s[i+$Nx+1] - 2*$s[i] + $s[i-$Nx-1]) / dy^2)
                    
                elseif @capture(tmp, Differential(x)(s_(x, y)))
                    return :(($s[i+1] - $s[i-1]) / (2*dx))
                    
                elseif @capture(tmp, Differential(y)(s_(x, y)))
                    return :(($s[i+$Nx+1] - $s[i-$Nx-1]) / (2*dy))
                    
                elseif @capture(tmp, s_(x, y))
                    return :($s[i])
                end
            else
                # 1D case
                if @capture(tmp, Differential(x)(Differential(x)(s_(x,))))
                    return :(($s[i+1] - 2*$s[i] + $s[i-1]) / dx^2)
                    
                elseif @capture(tmp, Differential(x)(s_(x,)))
                    return :(($s[i+1] - $s[i-1]) / (2*dx))
                    
                elseif @capture(tmp, s_(x,))
                    return :($s[i])
                end
            end
            tmp
        end
    end
    aux
end

function make_boundary_assignments_2D(vars, nvars)
    assignments = Expr[]
    for k in 1:nvars
        R_var = Symbol("R_", vars[k])
        var = vars[k]
        push!(assignments, :($R_var[i] = $var[i]))
    end
    return assignments
end


function make_boundary_assignments_1D(vars, nvars, conds, interior_eqs)
    left_assignments = Expr[]
    right_assignments = Expr[]
    
    for k in 1:nvars
        R_var = Symbol("R_", vars[k])
        var = vars[k]
        
        # Left boundary
        if haskey(conds, var) && conds[var][1] !== nothing
            left_bc = conds[var][1]
            push!(left_assignments, :($R_var[i] = $var[i] - $left_bc))
        else
            push!(left_assignments, :($R_var[i] = $(interior_eqs[k])))
        end
        
        # Right boundary
        if haskey(conds, var) && conds[var][2] !== nothing
            right_bc = conds[var][2]
            push!(right_assignments, :($R_var[i] = $var[i] - $right_bc))
        else
            # Zero-gradient (Neumann) boundary condition when no BC specified
            push!(right_assignments, :($R_var[i] = $var[i] - $var[i - 1]))
        end
    end
    
    return left_assignments, right_assignments
end

function make_interior_assignments(vars, eqs, nvars)
    assignments = Expr[]
    for k in 1:nvars
        R_var = Symbol("R_", vars[k])
        push!(assignments, :($R_var[i] = $(eqs[k])))
    end
    return assignments
end

function make_view_bindings(vars, nvars, N)
    bindings = Expr[]
    for k in 1:nvars
        var = vars[k]
        R_var = Symbol("R_", var)
        start_idx = (k-1)*N + 1
        end_idx = k*N
        push!(bindings, :($var = view(U, $start_idx:$end_idx)))
        push!(bindings, :($R_var = view(R, $start_idx:$end_idx)))
    end
    return bindings
end

function create_residual_function(eqs::Vector{Expr}, vars::Vector{Symbol}, Nx::Int, Ny::Int=0, conds::Dict=Dict())
    nvars = length(vars)
    is_2D = Ny > 0
    N = is_2D ? (Nx+1) * (Ny+1) : (Nx+1)
    
    view_bindings = make_view_bindings(vars, nvars, N)
    interior_assignments = make_interior_assignments(vars, eqs, nvars)

    if is_2D
        boundary_assignments = make_boundary_assignments_2D(vars, nvars)
    else
        left_assignments, right_assignments = make_boundary_assignments_1D(vars, nvars, conds, eqs)
    end
    
    
    if is_2D
        quote
            function residual!(R, U, p)
                dx, dy = p
                Nx = $Nx
                
                $(view_bindings...)
                
                for j in 1:$Ny+1
                    for ii in 1:$Nx+1
                        i = ii + (j-1) * ($Nx+1)
                        x = (ii-1) * dx
                        y = (j-1) * dy
                        
                        if ii == 1 || ii == $Nx+1 || j == 1 || j == $Ny+1
                            $(boundary_assignments...)
                        else
                            $(interior_assignments...)
                        end
                    end
                end
                return R
            end
        end
    else
        quote
            function residual!(R, U, p)
                dx = p[1]
                
                $(view_bindings...)
                
                for i in 1:$Nx+1
                    x = (i-1) * dx
                    
                    if i == 1
                        $(left_assignments...)
                    elseif i == $Nx+1
                        $(right_assignments...)
                    else
                        $(interior_assignments...)
                    end
                end
                return R
            end
        end
    end
end