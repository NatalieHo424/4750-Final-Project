using JuMP
using HiGHS
using Statistics
using Plots
using Distributions
using Random

Random.seed!(42)

# ----------------------------
# 0. SETTINGS (unchanged)
# ----------------------------
nT = 12
Q = 100_000.0
Cin = 40.0
Lin = Q * Cin / 1000.0
days = fill(365.0 / 12.0, nT)

T_base = [6.0,6.5,8.0,10.5,14.0,18.0,21.0,22.0,19.0,14.5,9.0,6.5]

eta_DNA_ref = 0.90
theta = 1.07
T_ref = 20.0
PNA_daily_limit = 10000.0

e_DNA = fill(4.0, nT)
e_PNA = fill(1.0, nT)
heat_energy_per_deg = 0.006
c_elec = 0.10
TN_target = 10.0
M_heat = 30.0

AnnualCapex_PNA = 365_000.0
EF_elec = 0.40
EF_N2O_N_per_kgN = 0.003
GWP_N2O = 298.0
EF_N2O_CO2e_per_kgN = EF_N2O_N_per_kgN * (44/28) * GWP_N2O
SCC = 80.0

# ----------------------------
# Monte Carlo settings
# ----------------------------
N_MC = 300
CV_T = 0.15   # coefficient of variation for temperature

# Storage
xPNA_MC = zeros(N_MC, nT)
heat_MC = zeros(N_MC, nT)
cost_MC = zeros(N_MC)

# ----------------------------
# Monte Carlo loop
# ----------------------------
for k in 1:N_MC

    # --- draw stochastic temperatures ---
    T_draw = [
        rand(LogNormal(log(T_base[m]) - 0.5*log(1+CV_T^2),
                       sqrt(log(1+CV_T^2))))
        for m in 1:nT
    ]

    # --- recompute efficiencies ---
    eta_base = [min(max(theta^(T_draw[m]-T_ref)*0.85,0.01),0.99) for m in 1:nT]
    Δ = 1.0
    slopes = [(min(max(theta^(T_draw[m]+Δ-T_ref)*0.85,0.01),0.99) -
               min(max(theta^(T_draw[m]-Δ-T_ref)*0.85,0.01),0.99)) / (2*Δ)
               for m in 1:nT]

    model = Model(HiGHS.Optimizer)

    @variables(model,
        begin
            0 <= x_DNA[m=1:nT] <= 1
            0 <= x_PNA[m=1:nT] <= 1
            0 <= heat_deg[m=1:nT] <= M_heat
            y_PNA, Bin
        end
    )

    @constraint(model, [m=1:nT], x_DNA[m] + x_PNA[m] == 1)
    @constraint(model, [m=1:nT], x_PNA[m] <= y_PNA)
    @constraint(model, [m=1:nT], x_PNA[m]*Lin <= PNA_daily_limit)

    @expression(model, R_DNA[m=1:nT], eta_DNA_ref*x_DNA[m]*Lin)
    @expression(model, R_PNA[m=1:nT], x_PNA[m]*Lin*eta_base[m] + slopes[m]*Lin*heat_deg[m])

    @expression(model, Ceff[m=1:nT], (Lin - R_DNA[m] - R_PNA[m])/Q*1000)
    @constraint(model, [m=1:nT], Ceff[m] <= TN_target)

    @expression(model, E_total[m=1:nT],
        e_DNA[m]*R_DNA[m] + e_PNA[m]*R_PNA[m] + heat_deg[m]*heat_energy_per_deg*Q)

    @expression(model, Cost_energy,
        sum(days[m]*c_elec*E_total[m] for m in 1:nT))

    @expression(model, Cost_GHG,
        sum(days[m] *
            (E_total[m]*EF_elec + R_DNA[m]*EF_N2O_CO2e_per_kgN) *
            (SCC/1000)
            for m in 1:nT))

    @objective(model, Min,
        y_PNA*AnnualCapex_PNA + Cost_energy + Cost_GHG)

    optimize!(model)

    xPNA_MC[k,:] .= value.(x_PNA)
    heat_MC[k,:] .= value.(heat_deg)
    cost_MC[k] = objective_value(model)
end

# ----------------------------
# Results & plots
# ----------------------------
xPNA_mean = mean(xPNA_MC, dims=1)[:]
heat_mean = mean(heat_MC, dims=1)[:]

println("\n=== MONTE CARLO RESULTS ===")
println("Mean annual cost: \$", round(mean(cost_MC), digits=0))
println("Std dev cost: \$", round(std(cost_MC), digits=0))

p1 = plot(1:12, xPNA_mean.*100, lw=2, marker=:circle,
    xlabel="Month", ylabel="% Flow to PNA",
    title="Monte Carlo Mean PNA Utilization")

p2 = plot(1:12, heat_mean, lw=2, marker=:square,
    xlabel="Month", ylabel="Heating (°C)",
    title="Monte Carlo Mean Heating")

plot(p1, p2, layout=(1,2), size=(900,400))
