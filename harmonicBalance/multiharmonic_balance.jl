using MethodOfLines
using NonlinearSolve
import ModelingToolkit as Model
import SymPy as sp
import Symbolics as Symb
using DomainSets
import ApproxFun as AF
import DifferentialEquations as DE
using Symbolics
using SparseArrays

function create_ansatz(coords::Tuple, t::Symbolics.Num, omega, harmonics::Int, n_fields::Int=1)
    var_names = Symbol[]
    var_exprs = Symbolics.Num[]
    fields = Symbolics.Num[]
    
    letter_idx = 1
    
    for field_idx in 1:n_fields
        u = Num(0)
        j = 1
        base_char = Char('A' + 2*(field_idx - 1))
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

function make_equations(expanded, harmonics, omega, t)
    eqs = []
    for i in 1:harmonics
        sin_coef = Symb.coeff(expanded, sin(i*omega*t))
        cos_coef = Symb.coeff(expanded, cos(i*omega*t))
        push!(eqs, sin_coef)
        push!(eqs, cos_coef)
    end
    return eqs
end


# Implementing with the custom stencil
using MacroTools: prewalk, @capture
using Symbolics: Differential

function transform_boundaries(equation, bound)
    is_2D = false
    
    function aux(ex)
        prewalk(ex) do tmp
            if is_2D
                # 2D case
                return ex
            else
                # 1D case
                if bound == 1
                    if @capture(tmp, s_[i-1])
                        return 0
                    end
                else 
                    if @capture(tmp, s_[i+1])
                        return 0
                    end
                end
            end
            tmp
        end
    end
    aux(equation)
end

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
        if haskey(conds, var) && isa(conds[var][1], Number)
            left_bc = conds[var][1]
            push!(left_assignments, :($R_var[i] = $var[i] - $left_bc))
        elseif haskey(conds, var) && conds[var][1] == "neumann"
            push!(left_assignments, :($R_var[i] = $var[i + 1] - $var[i]))
        elseif haskey(conds, var) && conds[var][1] == nothing
            boundary_eqs = transform_boundaries(interior_eqs[k], 1)
            
            push!(left_assignments, :($R_var[i] = $(boundary_eqs)))
        end
        
        # Right boundary
        if haskey(conds, var) && isa(conds[var][2], Number)
            right_bc = conds[var][2]
            push!(right_assignments, :($R_var[i] = $var[i] - $right_bc))
        elseif haskey(conds, var) && conds[var][2] == "neumann"
            # Zero-gradient (Neumann) boundary condition when no BC specified
            push!(right_assignments, :($R_var[i] = $var[i] - $var[i - 1]))
        elseif haskey(conds, var) && conds[var][2] == nothing
            boundary_eqs = transform_boundaries(interior_eqs[k], 2)

            push!(right_assignments, :($R_var[i] = $(boundary_eqs)))
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


function create_jac_blocks(equations, amplitudes, harmonicsNum)
    DiffMat = Vector{Union{Expr, Float64, Int64}}(undef, (2*harmonicsNum)^2)
    LaplCoeff = Vector{Union{Expr, Float64, Int64}}(undef, ((2*harmonicsNum)^2)*2)
    for i in 1:2*harmonicsNum
        for j in 1:2*harmonicsNum
            index = (i-1)*2*harmonicsNum + j
            temp = Meta.parse(string(expand_derivatives(Differential(amplitudes[j])(equations[i]))))
            # println("___________________________")
            # println(temp)
            # println("___________________________")
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


function always_expr(s::String)
    e = Meta.parse(s)
    return isa(e, Expr) ? e : :($e)
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