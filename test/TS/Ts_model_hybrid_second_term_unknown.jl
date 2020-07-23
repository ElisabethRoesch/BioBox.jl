# Do same as static but with adding ODE sol each time new.
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0_network = [0.001; 0.001]
u0 = Float32[3., 2.9]
datasize = 30
tspan = (0.0f0, 20.f0)
ps = [100., 2., 100., 2.] # from Supplementary material

# derivative
function trueODEfunc(states, ps, t)
    alpha1,beta,alpha2, gamma = ps
    u,v = states
    du=alpha1/(1+v^beta)-u
    dv=alpha2/(1+u^gamma)-v
    return [du, dv]
end

t = range(tspan[1], tspan[2], length = datasize)
prob = ODEProblem(trueODEfunc, u0, tspan, ps)
ode_data = Array(solve(prob, Tsit5(),saveat = t))
pl = scatter(t,ode_data[1,:],label = "Complete dynamics of U", xlabel = "Time", ylabel = "Species abundance", grid = "off")
scatter!(t,ode_data[2,:],label = "Complete dynamics of V")
function knownPartODEfunc(states, ps, t)
  alpha1,beta,alpha2, gamma = ps
  u,v = states
  du = -u
  dv = -v
  return [du, dv]
end
knownProb = ODEProblem(knownPartODEfunc, u0, tspan, ps)
knownPartData = Array(solve(knownProb, Tsit5(), saveat = t))

pl = plot(t,ode_data[1,:], label = "Observed data", xlabel="Time", ylabel="Species abundance", grid = "off")
plot!(t,ode_data[2,:], label = "Observed data")
scatter!(t,knownPartData[1,:], label = "Known dynamics of U")
scatter!(t,knownPartData[2,:], label = "Known dynamics of V")
#savefig("plots/prior4.pdf")
dudt2 = FastChain(FastDense(2, 50, tanh),
            FastDense(50, 2))
n_ode = NeuralODE(dudt2, tspan, Tsit5(), saveat = t)

function predict_n_ode(p)
  current_network = n_ode.model # get current NN of Neural ODE
  print(current_network(u0_network, n_ode.p)) # get gradient prediction for a U of one time point (here u0_network)
  n_ode(u0_network,p).+knownPartData
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,pred .- ode_data)
    loss,pred
end

loss_n_ode(n_ode.p) # n_ode.p stores the initial parameters of the neural ODE

cb = function (p, l, pred; doplot = false) #callback function to observe training
  display(l)
  if doplot
    pl = scatter(t, ode_data[1,:],label = "Observed data", xlabel = "Time", ylabel = "Species abundance", grid = "off",show = true)
    plot!(t, pred[1,:], label = "Prediction")
    scatter!(t, ode_data[2,:], label = "Observed data")
    plot!(t, pred[2,:], label = "Prediction")
    display(plot(pl))
  end
  return false
end
# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...; doplot = true)
res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.01), cb = cb, maxiters = 2000)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...; doplot = true)
#savefig("plots/test_result_hybrid_model4.pdf")
