


function create_jacobian_function(N, DiffMat, LaplCoeff)
    function_code = quote
        function jacobian!(J, U, p)
            dx, dy = p
            grid_size = $N * $N

            A_array = @view(U[1:grid_size])
            B_array = @view(U[grid_size+1:2*grid_size])

            Jmat = spzeros(2*grid_size, 2*grid_size)

            #Boundary conditions (easier to write over all diagonal points, then overwrite later):
            for k in 1:2*grid_size
                Jmat[k,k] = 1
            end

            #Inner points:
            for i in 2:$(N-1)
                for j in 2:$(N-1)
                    idx = (j-1)*$N + i
                    Jmat[idx, idx] = $(DiffMat[1][1]) - 2*LaplCoeff[1][1]/(dx^2) - 2*LaplCoeff[1][2]/(dy^2)
                    Jmat[idx, idx  + grid_size] = $(DiffMat[1][2])
                    Jmat[idx + grid_size, idx] = $(DiffMat[2][1])
                    Jmat[idx + grid_size, idx + grid_size] = $(DiffMat[2][2]) - 2*LaplCoeff[2][1]/(dx^2) - 2*LaplCoeff[2][2]/(dy^2)

                    Jmat[idx, idx - 1] = LaplCoeff[1][1]/(dx^2)
                    Jmat[idx, idx + 1] = LaplCoeff[1][1]/(dx^2)
                    Jmat[idx, idx - $N] = LaplCoeff[1][2]/(dy^2)
                    Jmat[idx, idx + $N] = LaplCoeff[1][2]/(dy^2)

                    Jmat[idx + grid_size, idx + grid_size - 1] = LaplCoeff[2][1]/(dx^2)
                    Jmat[idx + grid_size, idx + grid_size + 1] = LaplCoeff[2][1]/(dx^2)
                    Jmat[idx + grid_size, idx + grid_size - $N] = LaplCoeff[2][2]/(dy^2)
                    Jmat[idx + grid_size, idx + grid_size + $N] = LaplCoeff[2][2]/(dy^2)
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