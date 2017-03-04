using StructuralEstimation
# Set RNG seed
srand(124)

# State space
n = 500
o_max = 450
Xval = linspace(0, o_max, n)
p = [0.0937; 0.4475; 0.4459; 0.0127; 0.0002];

F1 = zeros(n, n)
offset = -1
for i = 1:n, j = 1:length(p)
    i+offset+j > n && continue
    F1[i,i+offset+j] = i+offset+j == n ? sum(p[j:end]) : p[j]
end

# We can handle sparse matrices
sparse_F = [sparse(F1), sparse(F1)[ones(Int64, n), :]]
F = [F1, F1[ones(Int64, n), :]]

S = State(Xval, F)
sparse_S = State(Xval, sparse_F)
bdisc = 0.9999
Z1 = [zeros(n) -0.001*Xval]
Z2 = [-ones(n) zeros(n)]

U = LinearUtility([Z1, Z2], bdisc, [11.;2.5])
V, iters = solve!(U,S)
T = 100
N = 100

D = simulate(U, S, 1, T, N)

#ESTIMATE TRANSITIONS

U.Θ*=0.
rust_nfxp = fit_nfxp(U, S, D)
rust_npl = fit_npl(U, S, D)

U = LinearUtility([Z1, Z2], 0.9999, [11., 3.]*0)
@time rust_nfxp = fit_nfxp(U, S, D; verbose = false);
U = LinearUtility([Z1, Z2], 0.9999, [11., 3.]*0)
@time rust_npl = fit_npl(U, S, D)
