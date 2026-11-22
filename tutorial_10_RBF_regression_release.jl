using Plots
using Distributions, Random, Statistics
using LinearAlgebra, SparseArrays, JSON
cd(dirname(@__FILE__))

using Colors
purple=RGB(122/256,30/256,71/256)
blue=RGB(0/256, 39/256, 76/256)
mygreen=RGB(83/256,165/256,132/256)


# Load wind power records 
dataset = JSON.parsefile("wind_power_data.json")
n = length(dataset["ActivePower"])
wind_data = Dict(:w => dataset["ActivePower"], :x => dataset["WindSpeed"]./10)

# Load wind power records 
dataset = JSON.parsefile("wind_power_data.json")
n = length(dataset["ActivePower"])
sort_index = sortperm(dataset["WindSpeed"])
t = dataset["ActivePower"][sort_index]
x = dataset["WindSpeed"][sort_index]/10


# Gaussian radial basis functions (RBF)
# k number of RBFs, each centered at c[k], scale γ = 5
set_RBF=Dict(:c => [i for i in 0.0:0.05:1.1], :γ => 5, :k => 0)
set_RBF[:k] = length(set_RBF[:c])
# RBF definition 
φ(x,j) = exp(-(set_RBF[:γ]*norm(x-set_RBF[:c][j],2))^2)

# stochastic gradient descent (SGD)
# initialize parameters 
Random.seed!(128)
w_0 = rand(); w = rand(set_RBF[:k]);
Random.seed!()
# settings 
α = 0.01; # step size
iter_max = 25000; # iteration limit
λ = 0.001; # regularization coefficient (default 0.001)
reg_type = "l2"

function my_SGD(w_0,w)
    for o in 1:iter_max
        # pick a random index of the sample from a dataset 
        i = sample(1:n, 1; replace=true)[1]

        # make a prediction of wind power output on the random sample 
        # this shold include an intercept w_0 followed by the weigted sum of RBFs φ(x,j)
        ŵ = [YOU CODE GOES HERE] 

        # update regression parameters without using regularization (SLIDE 25)
        # update the intersept 
        w_0 = [YOU CODE GOES HERE]
        # update the weigths of RBFs 
        for l in 1:set_RBF[:k]
            w[l] = [YOU CODE GOES HERE]
        end

        # account for regularization (SLIDE 27 and 29)
        if reg_type == "l2"
            w_0 -= [YOU CODE GOES HERE]
            w .-= [YOU CODE GOES HERE]
        end
        if reg_type == "l1"
            w_0 = sign(b) .* max(0,abs(w_0)-α*λ)
            w .= sign.(w) .* max.(0,abs.(w).-α*λ)
        end   
    end
    return w_0, w
end

@time w_0, w = my_SGD(w_0,w)


# to set intercept to 0 and weights to 1s to see unoptimized RBFs
# w_0 = 0
# w = ones(set_RBF[:k])

# make a function to make prediction on x 
w̅(x) = w_0 + sum(w[j]*φ(x,j) for j in 1:set_RBF[:k])

# make a plot
plo = scatter(x,t,c=blue,frame=:box,label=false,xlabel="normalized wind speed",ylabel="power output [p.u.]")
plot!(ylims=(-0.25,1.2),xlims=(0.25,1))
for i in 1:set_RBF[:k]
    plot!(x, w[i]*φ.(x,i),c=:gray,label=false,linestyle=:dash)
end
plot!(x, ones(length(x))*w_0,c=:gray,label=false,linestyle=:dash)
plot!(w̅,0.25,1,lw=5,c=purple,label=false)
