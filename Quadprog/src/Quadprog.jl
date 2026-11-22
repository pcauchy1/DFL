module Quadprog
# originally written by Stefan Harmeling, 2014-11-12
# updated for Julia 1.x and Ipopt.jl 1.x

export quadprog

using LinearAlgebra
using Ipopt

# convert scalars, vectors, adjoints, and Int arrays to a plain Vector{Float64}
tovec(x::Number)        = [Float64(x)]
tovec(x::AbstractArray) = Float64.(vec(x))

function quadprog(Q, c;
                  A   = zeros(0, length(c)),
                  b   = zeros(0),
                  Aeq = zeros(0, length(c)),
                  beq = zeros(0),
                  lb  = fill(-Inf, length(c)),
                  ub  = fill(Inf,  length(c)),
                  x0  = zeros(length(c)))

    # min  0.5 * x' * Q * x + c' * x
    # s.t. A   * x <= b
    #      Aeq * x == beq
    #      lb <= x <= ub
    # init x with x0

    # normalize types
    Q   = Matrix{Float64}(Q)
    c   = tovec(c)
    A   = Matrix{Float64}(A)
    Aeq = Matrix{Float64}(Aeq)
    b   = tovec(b)
    beq = tovec(beq)
    lb  = tovec(lb)
    ub  = tovec(ub)
    x0  = tovec(x0)

    n   = length(x0)       # number of variables
    m   = size(A,   1)     # number of inequality constraints
    meq = size(Aeq, 1)     # number of equality constraints

    issymmetric(Q) || error("Q must be symmetric")
    size(Q) == (n, n)                    || error("size mismatch: Q")
    length(c) == n                       || error("size mismatch: c")
    length(b) == m                       || error("size mismatch: b")
    length(beq) == meq                   || error("size mismatch: beq")
    size(A, 2) == n                      || error("size mismatch: A")
    size(Aeq, 2) == n                    || error("size mismatch: Aeq (needs to be meq × n)")
    (length(lb) == n && length(ub) == n) || error("size mismatch: lb/ub")

    # constraint bounds: first m rows are A*x <= b, then meq equalities
    g_L = vcat(fill(-Inf, m), beq)
    g_U = vcat(b, beq)
    Ac  = vcat(A, Aeq)     # stacked (m+meq) × n constant Jacobian

    eval_f(x) = 0.5 * dot(x, Q * x) + dot(c, x)

    function eval_grad_f(x, grad_f)
        grad_f .= Q * x .+ c
        return
    end

    function eval_g(x, g)
        g .= Ac * x
        return
    end

    function eval_jac_g(x, rows, cols, values)
        if values === nothing
            # sparsity structure (1-based, column-major order of Ac)
            k = 1
            for j in 1:n, i in 1:(m + meq)
                rows[k] = i
                cols[k] = j
                k += 1
            end
        else
            values .= vec(Ac)
        end
        return
    end

    function eval_h(x, rows, cols, obj_factor, lambda, values)
        # Ipopt wants only the lower triangle of the symmetric Hessian
        if values === nothing
            k = 1
            for i in 1:n, j in 1:i
                rows[k] = i
                cols[k] = j
                k += 1
            end
        else
            k = 1
            for i in 1:n, j in 1:i
                values[k] = obj_factor * Q[i, j]
                k += 1
            end
        end
        return
    end

    prob = Ipopt.CreateIpoptProblem(
        n, lb, ub,
        m + meq, g_L, g_U,
        (m + meq) * n,        # nonzeros in Jacobian
        n * (n + 1) ÷ 2,      # nonzeros in Hessian (lower triangle)
        eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h,
    )
    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    Ipopt.AddIpoptNumOption(prob, "tol", 1e-12)
    prob.x = x0
    status = Ipopt.IpoptSolve(prob)
    return (copy(prob.x), prob.obj_val, status)
end

end # module