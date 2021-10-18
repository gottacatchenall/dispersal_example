using Weave
using DynamicGrids, Dispersal
using NeutralLandscapes
using SimpleSDMLayers
using Colors, ColorSchemes
using Random
using Plots

Random.seed!(2) # 300 

dims = (400, 400)
patches = convert(
    Matrix{Int32},
    broadcast(x -> x < 0.3 || x > 0.8, rand(DiamondSquare(0.9), dims...)),
)
heatmap(patches)

# simulated env variable that effects growth
totalenv = rand(DiamondSquare(0.7), dims...)
env = mask(SimpleSDMPredictor(patches), SimpleSDMPredictor(totalenv))
plot(env, aspectratio = 1, frame = :box, cbtitle = "environmental variable")


# density dependent local dispersal
struct DensityDependentDispersal{R,W,P,A,C,N,KT} <: SetNeighborhoodRule{R,W}
    baseprobability::P
    α::A
    K::C
    neighborhood::N
    kernel::KT
end

dispersalprobability(N, p, α, K) = p - 0.5 + 1 / (1 + exp(-α * N / K));

nvec = 0:0.01:200
plot(
    xlim = (0, 200),
    ylim = (0, 1),
    frame = :box,
    xlabel = "abundance (N)",
    ylabel = "dispersal probability",
)
plot!(nvec, dispersalprobability.(nvec, 0.1, 0.5, 100), label = "α=0.5")
plot!(nvec, dispersalprobability.(nvec, 0.1, 1, 100), label = "α=1")
plot!(nvec, dispersalprobability.(nvec, 0.1, 3, 100), label = "α=3")


function DynamicGrids.applyrule!(
    data,
    rule::DensityDependentDispersal{R,W},
    N,
    I,
) where {R,W}
    isnothing(N) && return nothing
    N > zero(N) || return 0

    α, K, p = rule.α, rule.K, rule.baseprobability

    rand() < dispersalprobability(N, p, α, K) || return N
    sum = zero(N)
    for (offset, k) in zip(offsets(rule.kernel), kernel(rule.kernel))
        @inbounds propagules = N * k
        @inbounds add!(data[W], propagules, I .+ offset...)
        sum += propagules
    end
    @inbounds sub!(data[W], sum, I...)
    return N - sum
end

struct EnvironmentalMediatedLogisticGrowth{R,W,L,C,E,S,F} <: SetCellRule{R,W}
    λ::L
    K::C
    envgrid::E
    σ::S
    envfunction::F
end
EnvironmentalMediatedLogisticGrowth(
    λ,
    grid;
    K = 200,
    σ = 0.2,
    f = (x, σ) -> exp((-(0.5 - x) * σ)^2),
) = EnvironmentalMediatedLogisticGrowth(λ, K, grid, σ, f)

function DynamicGrids.applyrule!(data, rule::EnvironmentalMediatedLogisticGrowth, N, I)
    isnothing(N) && return nothing
    N > zero(N) || return zero(N)
    λ, K, σ, envlayer = rule.λ, rule.K, rule.σ, rule.envgrid
    env = envlayer[I...]

    isnothing(env) && return nothing

    Δt = 0.0001
    fitness = rule.envfunction(env, σ)
    realizedgrowthrate = fitness * λ * (1 / Δt)

    if realizedgrowthrate > zero(realizedgrowthrate)
        return @fastmath (N * K) / (N + (K - N) * exp(-realizedgrowthrate))
    else
        return @fastmath N * exp(realizedgrowthrate)
    end
end


localdispersalprob = 0.01
density_dependence = 3
carrying_capacity = 100
mean_intrisic_growth_rate = 2.5
environment_selection_strength = 0.1
long_range_dispersal_probability = 0.005
long_range_dispersal_distance = 150
dispersalneighborhood = VonNeumann(2)

localdispersal = DensityDependentDispersal(
    localdispersalprob,
    density_dependence,
    carrying_capacity,
    dispersalneighborhood,
    DispersalKernel(neighborhood = dispersalneighborhood),
)
jumpdispersal =
    JumpDispersal(long_range_dispersal_probability, long_range_dispersal_distance)
localgrowth = EnvironmentalMediatedLogisticGrowth(
    mean_intrisic_growth_rate,
    env;
    σ = environment_selection_strength,
    K = carrying_capacity,
)


init = zeros(Float32, size(patches))
init[190:200, 1] .= 100 # start invasion from left

# choose a random start index
rules = Ruleset(localgrowth, localdispersal, jumpdispersal)

cs = ColorSchemes.thermal
 # dg.jl flips the output for some reason so we flip the mask and init
patchmask = Bool.(patches)[end:-1:1, 1:end]

# setup gif output 
output = GifOutput(
    init[end:-1:1, 1:end],
    filename = "./dispersal.gif",
    mask = patchmask,
    scheme = cs,
    tspan = 1:500,
    fps = 25,
    minval = 0,
    maxval = 5,
)

sim!(output, rules)
