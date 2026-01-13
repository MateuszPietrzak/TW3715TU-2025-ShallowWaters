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
N = 50
harmonics = 1
stepx = (xright-xleft)/(N-1)
stepy = (yright-yleft)/(N-1)

#Definition of constants (placeholders):
gamma::Float64 = 10.0
gamma_3::Float64 = 1.0
c::Float64 = 10.0
omega::Float64 = 1

@parameters x, y, t
@variables A1(..), B1(..)
r1 = @rule cos(~x)^3 => 0.75 * cos(~x) + 0.25 * cos(3 * ~x)
r2 = @rule sin(~x)^3 => 0.75 * sin(~x) - 0.25 * sin(3 * ~x)
r3 = @rule cos(~x)^2 => 1 - sin(~x)^2
r4 = @rule sin(~x)^2 => 1 - cos(~x)^2

u = A1(x, y) * sin(omega*t) + B1(x, y) * cos(omega*t)
Dx = Differential(x)
Dy = Differential(y)
Dt = Differential(t)
y_eq = Dt(Dt(u)) - c*c*(Dx(Dx(u)) + Dy(Dy(u))) + gamma*Dt(u) + gamma_3*Dt(u)*Dt(u)*Dt(u)
y_exp = expand_derivatives(y_eq)

y_exp = simplify(expand(y_exp), RuleSet([r1, r2, r3, r4]))
y_exp = expand(y_exp)
y_exp = simplify(y_exp, RuleSet([r1, r2, r3, r4]))

sin_coeff = Symbolics.coeff(y_exp, sin(omega*t))
cos_coeff = Symbolics.coeff(y_exp, cos(omega*t))

#println("Sin coeff: ", sin_coeff)

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
        end
        return instr
    end
end

function transform_sym_coeff(ex)
    return prewalk(ex) do instr
        if @capture(instr, s_(x, y))
            id = string(s)
            letter = id[1]
            number = parse(Int, id[2:end])
            return :($(Symbol("$(letter)_array"))[i, j, $(number)])
        end
        return instr
    end
end

function create_residual_function(N, equationsExpr, harmonics)
    equationsExprMapped::Vector{Expr} = []

    println(length(equationsExpr))
    println(harmonics)

    for H in 1:harmonics
        if H == 1
            #F_view[2*H - 1][i, j] = $(equationsExpr[2*H - 1]) - 1000 * c^2 * exp(-40*(i * dx)^2)
            push!(equationsExprMapped, :(F_view[i, j,1,$(H)] = $(equationsExpr[2*H - 1]) - 1000 * c^2 * exp(-40*(i * dx)^2))) # Related to equationsExpr[2*H-1]
        else
            #F_view[2*H - 1][i, j] = $(equationsExpr[2*H - 1])
            push!(equationsExprMapped, :(F_view[i, j,1,$(H)] = $(equationsExpr[2*H - 1]))) # Related to equationsExpr[2*H-1]
        end
        #F_view[2*H][i, j] = $(equationsExpr[2*H])
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


function create_jac_blocks(equations, amplitudes, harmonicsNum)
    DiffMat = Vector{Expr}(undef, (2*harmonicsNum)^2)
    LaplCoeff = Vector{Union{Expr,Float64}}(undef, ((2*harmonicsNum)^2)*2)
    for i in 1:2*harmonicsNum
        for j in 1:2*harmonicsNum
            index = (i-1)*2*harmonicsNum + j
            DiffMat[index] = transform_sym_coeff(Meta.parse(string(expand_derivatives(Differential(amplitudes[j])(equations[i])))))
            LaplCoeff[index*2 - 1] = transform_sym_coeff(Meta.parse(string(Symbolics.coeff(equations[i], Differential(x)(Differential(x)(amplitudes[j]))))))
            LaplCoeff[index*2] = transform_sym_coeff(Meta.parse(string(Symbolics.coeff(equations[i], Differential(y)(Differential(y)(amplitudes[j]))))))
        end
    end
    return DiffMat, LaplCoeff
end


function create_jacobian_function(N, DiffMat, LaplCoeff, harmonics)
    equationsExprMapped::Vector{Expr} = []

    for HEq in 0:2*harmonics-1
        for HHar in 1:2*harmonics
            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size] = $(DiffMat[2*HEq*harmonics+HHar]) - 2*$(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2) - 2*$(LaplCoeff[2*(2*HEq*harmonics+HHar)])/(dy^2)))

            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size - 1] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))

            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size + 1] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)-1])/(dx^2)))

            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size - $(N)] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)])/(dy^2)))

            push!(equationsExprMapped, :(Jmat[idx + $(HEq)*grid_size, idx + $(HHar-1)*grid_size + $(N)] = $(LaplCoeff[2*(2*HEq*harmonics+HHar)])/(dy^2)))
        end
    end

    println(equationsExprMapped)

    function_code = quote
        function jacobian!(J, U, p)
            dx, dy = p
            grid_size = $N * $N
            harmonicsNum = $(harmonics)

            A_array = reshape(@view(U[1:harmonicsNum*grid_size]), $N, $N, harmonicsNum)
            B_array = reshape(@view(U[harmonicsNum*grid_size+1:2*harmonicsNum*grid_size]), $N, $N, harmonicsNum)

            Jmat = spzeros(2*grid_size*harmonicsNum, 2*grid_size*harmonicsNum)

            #Boundary conditions (easier to write over all diagonal points, then overwrite later):
            for k in 1:2*grid_size
                Jmat[k,k] = 1
            end

            #Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    idx = (j-1)*$N + i
                    $(equationsExprMapped...)
                end
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

equations = (sin_coeff, cos_coeff)
amplitudes = (A1(x, y), B1(x, y))
DiffMat, LaplCoeff = create_jac_blocks(equations, amplitudes, harmonics)


sin_ex = transform_sym(Meta.parse(string(sin_coeff)))
cos_ex = transform_sym(Meta.parse(string(cos_coeff)))

equationsExpr = (sin_ex, cos_ex)


println(sin_coeff)

println(DiffMat[1])
println(DiffMat[2])
println(DiffMat[3])
println(DiffMat[4])

println(LaplCoeff[3])


nonlinear_residual = create_residual_function(N, equationsExpr, harmonics)
initial_guess = ones(2*N*N)
par = [stepx, stepy]
jacobian = create_jacobian_function(N, DiffMat, LaplCoeff, harmonics)

nonlinear_function = NonlinearFunction(nonlinear_residual, jac=jacobian)
# nonlinear_function = NonlinearFunction(nonlinear_residual)

prob = NonlinearProblem(nonlinear_function, initial_guess, par)
println("Starting nonlinear solver...")
@time sol = NonlinearSolve.solve(prob, NewtonRaphson(), reltol = 1e-5, abstol = 1e-5)
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
            clims=(-7.5, 7.5),
            aspect_ratio=1)
end

gif(anim, "wave_jac.gif", fps=25)

