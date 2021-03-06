using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

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
savefig("plots/noprior.pdf")
dudt2 = FastChain((x,p) -> x.^3,
            FastDense(2,50,tanh),
            FastDense(50,2))
n_ode = NeuralODE(dudt2,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data  .- pred)
    loss,pred
end

loss_n_ode(n_ode.p) # n_ode.p stores the initial parameters of the neural ODE

cb = function (p,l,pred;doplot=false) #callback function to observe training
  display(l)
  # plot current prediction against data
  if doplot
    pl = scatter(t,ode_data[1,:],label="data")
    scatter!(pl,t,pred[1,:],label="prediction")
    scatter!(t,ode_data[2,:],label="data")
    scatter!(pl,t,pred[2,:],label="prediction")
    display(plot(pl))
  end
  return false
end
# Display the ODE with the initial parameter values.
cb(n_ode.p,loss_n_ode(n_ode.p)...;doplot=true)

res1 = DiffEqFlux.sciml_train(loss_n_ode, n_ode.p, ADAM(0.01), cb = cb, maxiters = 5000)
cb(res1.minimizer,loss_n_ode(res1.minimizer)...;doplot=true)

savefig("plots/test_result_ML_model.pdf")
