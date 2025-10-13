using ModelingToolkit, MethodOfLines, DifferentialEquations, Plots, DomainSets


xleft::Float64 = 0.0
xright::Float64 = 1.0
t0::Float64 = 0.0
T::Float64 = 4.0
N = 100
order = 2
step = (xright - xleft)/N
grid = collect(xleft:step:xright)

#Definition of constants (placeholders):
gamma::Float64 = 1
gamma_3::Float64 = 1
c::Float64 = 1
omega_d::Float64 = 1

#Define parameters and variables (note to self: u(..) is valid as a symbolic function notation):
@parameters x t
@variables u(..)

#Define differential operators:
Dt = Differential(t)
Dx = Differential(x)
Dtt = Differential(t)^2
Dxx = Differential(x)^2

#Define source function (placeholder constant spatial term; hell if I know which function we'll use):
source_f(x, t) = 1.0* sin(omega_d*t)

#Define initial condition (placeholder null value):
u0(x, t) = 0.0

#Define equation (kill me please):

equation = [Dtt(u(x,t)) + gamma*Dt(u(x,t)) + gamma_3*(Dt(u(x,t)))^3 ~ c^2*Dxx(u(x,t)) + source_f(x, t)]


#Define domains (kill me please x2, why do we need non standard symbols???):
domains = [x ∈ IntervalDomain(xleft, xright), t ∈ IntervalDomain(t0, T)]

#Define boundary conditions:
bound_conditions = [u(x, 0) ~ u0(x, 0),
                    u(xleft, t) ~ 0,
                    u(xright, t) ~ 0,
                    Dt(u(x, 0)) ~ 0]

#Create PDE system and assign identifier (no idea if order matters, I assume it does: eq, bcs, domains, paramers, variables):
@named pde_sys = PDESystem(equation, bound_conditions, domains, [x, t], [u(x,t)])

#Define a discretization (which doesn't discretize yet, for whatever reason):
discretization = MOLFiniteDifference([x=>grid], t, approx_order=order)

#Proceed with discretization:
@time eq_discretized = discretize(pde_sys,discretization)

#Solve the resulting system:
@time solution = DifferentialEquations.solve(eq_discretized, Trapezoid(), saveat=0.1)



#Plot results:
discrete_t = solution.t

sol_u = solution[u(x, t)]

Nx = length(grid)
Nt = length(solution.t)

animation = @animate for k in 1:Nt
    plot(grid, sol_u[1:end, k],
         title="t = $(solution.t[k])", xlabel="x", ylabel="u(x,t)")
end

gif(animation, "Example_Equation.gif", fps = 5)
