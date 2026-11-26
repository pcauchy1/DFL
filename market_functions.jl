using JuMP, Ipopt
using LinearAlgebra

function day_ahead_dispatch(h)

    P = 2 * [0.11  0      0;
             0     0.085  0;
             0     0      0.1225]

    Q = [5, 1.2, 1]
    A = [0, 0, 0]
    B = [250, 300, 270]
    G = [1, 1, 1]

    model = JuMP.Model(Ipopt.Optimizer)

    @variable(model, x[1:3])

    @constraint(model, A .<= x .<= B)
    @constraint(model, G' * x == h)

    @objective(model, Min, (1/2) * x' * P * x + Q' * x)

    set_silent(model)
    optimize!(model)

    generation = value.(x)
    total_cost = objective_value(model)

    return generation, total_cost
end

function real_time_dispatch(p_DA, Pg_RT; kup = 1.0, kdown = 1.0)

    B = [250, 300, 270]

    a = 2 * [0.11, 0.085, 0.1225]
    c = [5.0, 1.2, 1.0]

    P = [Diagonal(a)   -1 .* Diagonal(a);
        -1 .* Diagonal(a)   Diagonal(a)]

    Q = vcat(a .* p_DA .+ c .+ kup, a .* p_DA .- c .+ kdown)

    model = JuMP.Model(Ipopt.Optimizer)

    @variable(model, x[1:6] >= 0)t
    @constraint(model, [i=1:3], x[i] <= B[i] - p_DA[i])
    @constraint(model, [i=1:3], x[i+3] <= p_DA[i])
    @constraint(model, sum(x[1:3]) - sum(x[4:6]) == Pg_RT - sum(p_DA))

    @objective(model, Min, (1/2) * x' * P * x + Q' * x)

    set_silent(model)
    optimize!(model)

    upward = value.(x[1:3])
    downward = value.(x[4:6])
    total_cost = objective_value(model)

    return upward, downward, total_cost
end