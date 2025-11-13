function transform_sym(expr_str)
    corrected = string(expr_str)

    repl_lapl = [
        "Differential(x)(Differential(x)(A(x)))" => "(A[i+1] - 2A[i] + A[i-1]) / dx^2",
        "Differential(x)(Differential(x)(B(x)))" => "(B[i+1] - 2B[i] + B[i-1]) / dx^2",
        ]
    repl_first_der = [
        "Differential(x)(A(x))" => "(A[i+1] - A[i-1]) / (2*dx)",
        "Differential(x)(B(x))" => "(B[i+1] - B[i-1]) / (2*dx)",
        ]
    repl_linear = [
        "A(x)" => "A[i]",
        "B(x)" => "B[i]"
        ]

    corrected = replace(corrected, repl_lapl...)
    corrected = replace(corrected, repl_first_der...)
    corrected = replace(corrected, repl_linear...)

    repl_powers = [
        "A[i]^" => "(A[i])^",
        "B[i]^" => "(B[i])^"
        ]
    repl_mult_pattern = r"(?<=\d)(?=[A-Za-z\(])"

    corrected = replace(corrected, repl_powers...)
    corrected = replace(corrected, repl_mult_pattern => "*")

    #Finar corrections (required for scope):
    repl_sym = [
        "A" => "A_array",
        "B" => "B_array",
        "i" => "i_local",
        ]

    corrected = replace(corrected, repl_sym...)

    return corrected
end

function create_residual_function_1D(N, sin_eq_str, cos_expr_str)
    function_code = """
    function residual!(F, U, p)
        dx = p

        A_array = @view(U[1:$N+1])
        B_array = @view(U[$N+2:end])
        F_A = @view(F[1:$N+1])
        F_B = @view(F[$N+2:end])

        # Inner points:
        for i_local in 2:$(N)
            F_A[i_local] = ($sin_eq_str) - 100 * exp(-40(i_local * dx)^2)
            F_B[i_local] = ($cos_expr_str) 
        end

        # Dirichlet BCs
        F_A[1] = A_array[1]
        F_A[end] = A_array[end]
        F_B[1] = B_array[1]
        F_B[end] = B_array[end]

        return F
    end
    """
    return eval(Meta.parse(function_code))
end;