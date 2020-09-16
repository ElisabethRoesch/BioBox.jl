# Do same as static but with adding ODE sol each time new.
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
u0_network = [0.001;0.001]
u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,3.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
pl = scatter(t,ode_data[1,:],label="data")
scatter!(t,ode_data[2,:],label="data")
function knownPartODEfunc(du,u,p,t)
    # true_A = [-0.1 .0; -0.0 -0.1] # .pdf
    # true_A = [-0.0 2.0; -2.0 -0.0] # 2.pdf
    # true_A = [-0.1 0.0; -0.0 -0.0] # 3.pdf
    true_A = [-0.0 0.0; -0.0 -0.1] # 4.pdf

    du .= ((u.^3)'true_A)'
end
knownProb = ODEProblem(knownPartODEfunc,u0,tspan)
knownPartData = Array(solve(knownProb,Tsit5(),saveat=t))
pl = scatter(t,knownPartData[1,:],label="known part")
scatter!(t,knownPartData[2,:],label="known part")

pl = plot(t,ode_data[1,:],label="data")
plot!(t,ode_data[2,:],label="data")
scatter!(t,knownPartData[1,:],label="known part")
scatter!(t,knownPartData[2,:],label="known part")
#savefig("plots/prior4.pdf")
dudt2 = FastChain((x,p) -> x.^3, # Remove this line?
            FastDense(2, 50, tanh),
            FastDense(50, 2))
n_ode = NeuralODE(dudt2, tspan, Tsit5(), saveat = t)
function predict_n_ode(p)
  current_network = n_ode.model # get current NN of Neural ODE
  print(current_network(u0_network, n_ode.p)) # get gradient prediction for a U of one time point (here u0_network)
  test_call = knownPartODEfunc([0.0, 0.0], [0., 0.], 1, 1) # test cope of knownPartODEfunc
  #print(test_call)
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
  # plot current prediction against data
  if doplot
    pl = scatter(t,ode_data[1, :], label = "data")
    scatter!(pl,t,pred[1, :], label = "prediction")
    scatter!(t,ode_data[2, :], label = "data")
    scatter!(pl,t,pred[2, :], label = "prediction")
    display(plot(pl))
  end
  return false
end
# Display the ODE with the initial parameter values.
cb(n_ode.p, loss_n_ode(n_ode.p)...; doplot = true)

res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.01), cb = cb, maxiters = 500)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...; doplot = true)
#savefig("plots/test_result_hybrid_model4.pdf")
