# Verify paper claims directly from raw JSONL. Julia standard library only.
# Usage: julia verify_from_raw.jl

using JSON, Statistics

claims = JSON.parsefile("claims.json")

readjsonl(f) = [JSON.parse(line) for line in eachline(f)]

v1 = readjsonl("data/leakage_landscape_v1_final.jsonl")
v1_ok = filter(r -> get(r, "status", nothing) == "ok", v1)

v2 = readjsonl("data/leakage_landscape_v2.jsonl")

v3_an = readjsonl("data/v3/v3_an.jsonl")
v3_an_ok = filter(r -> get(r, "v3_status", nothing) == "ok", v3_an)

function dz(vals)
    x = filter(!isnan, vals)
    mean(x) / std(x)
end

function check(name, expected, got; tol=0.002)
    ok = abs(expected - got) <= tol
    println("  $name: expected=$expected, got=$(round(got, digits=4)), $(ok ? "PASS" : "FAIL")")
    ok
end

passed = 0; total = 0

println("=== Dataset counts ===")
global total += 1; global passed += check("n_datasets", claims["n_datasets"], length(v1_ok), tol=0)
global total += 1; global passed += check("corpus.median_n", claims["corpus"]["median_n"],
    median([r["n_rows"] for r in v1_ok]), tol=0)

println("\n=== Class I: Estimation ===")
d = [r["a_lr_gap_diff"] for r in v1_ok if haskey(r, "a_lr_gap_diff") && r["a_lr_gap_diff"] !== nothing]
global total += 1; global passed += check("norm_lr.dz", claims["norm_lr"]["dz"], dz(d))

println("\n=== Class II: Peeking ===")
d = [r["b_infl_k10"] for r in v1_ok if haskey(r, "b_infl_k10") && r["b_infl_k10"] !== nothing]
global total += 1; global passed += check("peek.dz", claims["peek"]["dz"], dz(d))
global total += 1; global passed += check("peek.auc", claims["peek"]["auc"], mean(d), tol=0.001)

println("\n=== Class II: Seed ===")
d = [r["ai_inflation"] for r in v2 if haskey(r, "ai_inflation") && r["ai_inflation"] !== nothing && !isnan(r["ai_inflation"])]
global total += 1; global passed += check("seed.dz", claims["seed"]["dz"], dz(d))

println("\n=== Class II: Screen ===")
d = [r["aq_k1_optimism"] for r in v2 if haskey(r, "aq_k1_optimism") && r["aq_k1_optimism"] !== nothing && !isnan(r["aq_k1_optimism"])]
global total += 1; global passed += check("screen.dz", claims["screen"]["dz"], dz(d))

println("\n=== N-scaling ===")
global total += 1; global passed += check("nscale.n_main", claims["nscale"]["n_main"],
    count(r -> get(r, "an_n_full", 0) == 2000, v3_an_ok), tol=0)
global total += 1; global passed += check("nscale.ext.n_datasets", claims["nscale"]["ext"]["n_datasets"],
    count(r -> get(r, "an_n_full", 0) == 10000, v3_an_ok), tol=0)

println("\n" * "="^40)
println("RESULT: $passed/$total checks passed")
passed == total && println("ALL CLAIMS VERIFIED FROM RAW DATA.")
