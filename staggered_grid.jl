using ModelingToolkit
using MethodOfLines
using DifferentialEquations
using DomainSets
using ProgressMeter
using Plots

@parameters t x
@variables η(..) u(..)
Dt = Differential(t);
Dx = Differential(x);

g = 9.81
H = 3
omega = 2*pi*3

x_l = 0
x_r = 2.0
tmax = 10.0
N = 10
dx = (x_r - x_l)/N
dt = dx/sqrt(g*H)
println("No. timesteps is")
println(tmax/dt)

IC(x) = x
F(x, t) = 3*sin(omega * t)
eqs = [Dt(η(t, x)) + H * Dx(u(t, x)) ~ 0,
       Dt(u(t, x)) + g * Dx(η(t, x)) ~ 0]

bcs = [η(0.0, x) ~ IC(x),
       u(0.0, x) ~ 0.0,
       Dx(u(t, x_r)) ~ 0.0,
       η(t, 0.0) ~ F(x, t)]

domains = [t in Interval(0.0, tmax),
           x in Interval(x_l, x_r)];

@named pdesys = PDESystem(eqs, bcs, domains, [t,x], [η(t,x), u(t,x)]);

discretization = MOLFiniteDifference([x=>dx], t, grid_align=MethodOfLines.StaggeredGrid(), edge_aligned_var=u(t,x));
println("dupa")
prob = discretize(pdesys, discretization, analytic = nothing);
println("dupa")
sol = solve(prob, ImplicitEuler(), dt=dt)


grid = x_l:dx:x_r
anim = @animate for t in range(0, tmax, length=200)
    u = sol[0](grid, t)
    plot(grid, u,
         ylim = (-1, 1),
         xlabel = "x",
         ylabel = "u(x,t)",
         title = "u(x, t),  t = $(round(t, digits=2))",
         legend = false)
end

# Save as GIF
gif(anim, "SWE_1D.gif", fps=30)
