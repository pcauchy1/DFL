using Plots
using JSON

using Colors
purple=RGB(122/256,30/256,71/256)
blue=RGB(0/256, 39/256, 76/256)
myred=RGB(220/256,20/256,60/256)
cd(dirname(@__FILE__))

# Load wind power records 
dataset = JSON.parsefile("wind_power_data.json")
n = length(dataset["ActivePower"])
sort_index = sortperm(dataset["WindSpeed"])
t = dataset["ActivePower"][sort_index]
x = dataset["WindSpeed"][sort_index]

# parameters for gradient descent 
α = 0.01 # learing rate 
iter_max = 5000

cost_function(w,b) = 1/(2n)*sum((w*x[i] + b - t[i])^2 for i in 1:n)

w, b = 0, 0
cost = zeros(iter_max)
for i in 1:iter_max
    ŷ = w .* x .+ b    
    w = # write weight update  
    b = # write bias update 
    cost[i] = cost_function(w,b)
    if i % 250 == 0
        println("GD iter $i ... prediction cost: $(round(cost_function(w,b),digits=6))")
    end
end

# plot gradient descent dynamics 
plot_gd = plot(cost,lw=5,c=purple,frame=:box,xlabel="iteration index",ylabel="cost function")

# plot wind power curve 
plot_wp = scatter(x,t,c=blue,frame=:box,xlabel="wind speed [m/s]",ylabel="power output [p.u.]",label="observations")
plot!(x,w .* x .+ b,lw=5,c=purple, label="prediction")
plot!(x,max.(w .* x .+ b,0),lw=2,c=myred, linestyle=:dot, label="projected")
