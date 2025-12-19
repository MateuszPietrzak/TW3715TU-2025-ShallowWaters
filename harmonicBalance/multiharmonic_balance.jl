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
            name = Symbol(Char('A' + letter_idx - 1))
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

function create_residual_function(eqs::Vector{Expr}, vars::Vector{Symbol}, Nx::Int, Ny::Int=0)
    nvars = length(vars)
    is_2D = Ny > 0
    N = is_2D ? (Nx+1) * (Ny+1) : (Nx+1)
    
    if is_2D
        quote
            function residual!(R, U, p)
                dx, dy = p
                Nx = $Nx
                
                $([:($(vars[k]) = view(U, $((k-1)*N+1):$(k*N))) for k in 1:nvars]...)
                $([:($(Symbol("R_", vars[k])) = view(R, $((k-1)*N+1):$(k*N))) for k in 1:nvars]...)
                
                for j in 1:$Ny+1
                    for ii in 1:$Nx+1
                        i = ii + (j-1) * ($Nx+1)
                        x = (ii-1) * dx
                        y = (j-1) * dy
                        
                        if ii == 1 || ii == $Nx+1 || j == 1 || j == $Ny+1
                            $([:($(Symbol("R_", vars[k]))[i] = $(vars[k])[i]) for k in 1:nvars]...)
                        else
                            $([:($(Symbol("R_", vars[k]))[i] = $(eqs[k])) for k in 1:nvars]...)
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
                
                $([:($(vars[k]) = view(U, $((k-1)*N+1):$(k*N))) for k in 1:nvars]...)
                $([:($(Symbol("R_", vars[k])) = view(R, $((k-1)*N+1):$(k*N))) for k in 1:nvars]...)
                
                for i in 1:$Nx+1
                    x = (i-1) * dx
                    
                    if i == 1 || i == $Nx+1
                        $([:($(Symbol("R_", vars[k]))[i] = $(vars[k])[i]) for k in 1:nvars]...)
                    else
                        $([:($(Symbol("R_", vars[k]))[i] = $(eqs[k])) for k in 1:nvars]...)
                    end
                end
                return R
            end
        end
    end
end