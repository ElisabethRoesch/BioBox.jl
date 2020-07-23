using Plots, DifferentialEquations

# Gardner switch (BMC Syst Biol 2016 Leon)
# u is the concentration of repressor 1
# v the concentration of repressor 2
# α1 and α2 denote the effective rates of synthesis of repressors 1 and 2
# β and γ are the cooperativity of repression of promoter 1 and of repressor 2

# derivative
function dudt(states, ps, t)
    alpha1,beta,alpha2, gamma =ps
    u,v=states
    du=alpha1/(1+v^beta)-u
    dv=alpha2/(1+u^gamma)-v
    return [du,dv]
end
# test setting
u0=[3.,2.9]
ps=[100.,2.,100.,2.] # from Supplementary material
tspan = (0.0,100.0)
prob =ODEProblem(dudt, u0, tspan, ps)

# solve example
sol = Array(solve(prob))

# visualise sol
plot(sol[1,:], label = "U")
plot!(sol[2,:], label = "V")
