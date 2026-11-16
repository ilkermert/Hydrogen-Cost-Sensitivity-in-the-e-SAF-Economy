Hydrogen Cost Sensitivity in the e-SAF Economy: A Stochastic LCOH-LCOP Linkage Model for Net-Zero Aviation in the EU
Authors: İlker Mert*, Hüseyin Yağlı

This repository contains representative code excerpts, the numerical input data and methodological documentation for the research article (under review):

"Hydrogen Cost Sensitivity in the e-SAF Economy: A Stochastic LCOH-LCOP Linkage Model for Net-Zero Aviation in the EU"

It follows the manuscript's own section order: each subsection below states the methodological component as described in the paper, immediately followed by a representative code excerpt implementing it.

Note on scope: These are representative excerpts of the core methodological steps, included for transparency and to show how each step was done. The complete analysis script is not included, see "Contact and Data Availability" below. The numerical input data of the study are provided in data.xlsx. This repository does not allow all tables and figures of the paper to be reproduced by running a script, and the paper does not claim that it does.

About the Study
The study propagates the uncertainty of the levelized cost of hydrogen (LCOH) to the levelized cost of production (LCOP) of e-SAF with a Monte Carlo simulation of 20,000 draws, applies it to an aggregate model and to ten EU countries, and scores the readiness of the countries with the Entropy Weight Method (EWM). LCOP is the cost per kilogram of total hydrocarbon product of a CO2-to-hydrocarbon pathway. No cost is allocated to the jet fraction or to coproducts. The projection to 2050 is scenario-based and conditional on the stated scenarios. The country ranking is conditional on the selected country set.

Stage	Content
core	Sampling, LCOH and LCOP, cost shares, correlation
country	Country-level LCOH and LCOP with national electricity price distributions
ewm	Readiness index, robustness to weights, normalisation, country subsets and perturbed indicators
sensitivity	PRCC, Sobol indices, scenarios, dependence scenarios, plant-scale sensitivity
horizon	Scenario-based projection for 2030 and 2050
reference	Reproduction of an independent reference case (Wolf et al., 2024)
2.1 Input Distributions and Sampling
All inputs are sampled from triangular distributions (minimum, mode, maximum) given in the sheet MC_Parameters of data.xlsx. One matrix of uniform random numbers is drawn once and reused wherever a comparison between countries or horizons is made (common random numbers), so differences between them do not come from sampling noise. Inputs are sampled independently in the baseline. Dependence is examined in separate scenarios.

def tri_ppf(u, a, m, b):
    c = (m - a) / (b - a)
    return np.where(u < c, a + np.sqrt(u * (b - a) * (m - a)), b - np.sqrt((1 - u) * (b - a) * (b - m)))

rng = np.random.default_rng(SEED)
U = rng.random((N, len(ORDER)))

def draw(U_, overrides=None):
    S = {p: tri_ppf(U_[:, col[p]], *param(p)) for p in ORDER}
    if overrides:
        S.update(overrides)
    return S
2.2 LCOH and LCOP
LCOH follows Equation 2 with CAPEX in USD/kW, specific electricity consumption in kWh/kg H2 and the electricity price in USD/MWh. LCOP follows Equation 3. The coefficients alpha and beta are effective feedstock requirements per kg of total hydrocarbon product. Their lower bounds are the stoichiometric minima of a jet-range n-alkane (C12H26), rounded to two decimals (0.44 and 3.10). The synthesis cost gamma excludes hydrogen and CO2 supply.

def model(S, capex_f=1.0, pe_f=1.0, dw=0.0, cf_add=0.0, co2p=None, h2_f=1.0):
    w = np.clip(S["wacc"] + dw, 0.01, 1.0)
    L = S["lifetime_years"]
    crf = w * (1 + w) ** L / ((1 + w) ** L - 1)
    cf = np.clip(S["capacity_factor"] + cf_add, 0.05, 0.95)
    eta = S["efficiency_kwh_per_kg"]
    capex_term = S["capex_usd_per_kw"] * capex_f * (crf + S["om_fraction"]) * eta / (8760 * cf)
    elec_term = S["elec_price_usd_per_mwh"] * pe_f / 1000 * eta
    lcoh = (capex_term + elec_term) * h2_f
    cp = S["co2_price_usd_per_t"] if co2p is None else co2p
    lcop = (S["h2_ratio_kg_per_kg_saf"] * lcoh + S["co2_ratio_kg_per_kg_saf"] * cp / 1000
            + S["synthesis_cost_usd_per_kg"])
    return lcoh, lcop, capex_term, elec_term
2.3 Country-Level Application
For each country the aggregate electricity price distribution is replaced by a triangular distribution defined by the minimum, median and maximum of the twelve monthly means of the 2025 day-ahead wholesale price (EUR/MWh, converted at 1.09 USD/EUR). The prices are wholesale prices. They exclude grid charges, taxes and procurement arrangements. All other draws are shared between the countries.

for _, r in country_elec.iterrows():
    a, m, b = (r["elec_price_min_eur_mwh"] * FX, r["elec_price_mode_eur_mwh"] * FX, r["elec_price_max_eur_mwh"] * FX)
    Sc = draw(U, {"elec_price_usd_per_mwh": tri_ppf(U[:, col["elec_price_usd_per_mwh"]], a, m, b)})
    h, p, _, _ = model(Sc)
2.4 Readiness Index (EWM)
The index uses six criteria: C1 (LCOP), C2 (grid carbon intensity), C4 (maturity-weighted PtL project score), C5 (policy code 0 to 2), C6 (GDP per capita in PPS) and C7 (renewable generation over demand). The legal criterion C3 is not included. The C4 score of a country is the sum over its projects of the status weight multiplied by (1 + ln(1 + capacity in kt H2/y)). The entropy weights describe the variation of a criterion in the selected dataset. They do not establish the causal importance of a criterion. Terms with a zero share are omitted (0 ln 0 = 0).

def ewm(df, cols, benefit, norm_type="minmax"):
    X = df[cols].astype(float)
    Z = pd.DataFrame(index=X.index)
    for c in cols:
        lo, hi = X[c].min(), X[c].max()
        Z[c] = (X[c] - lo) / (hi - lo) if c in benefit else (hi - X[c]) / (hi - lo)
    P = Z / Z.sum(axis=0)
    k = 1 / np.log(len(P))
    plogp = np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0.0)
    e = -k * plogp.sum(axis=0)
    d = 1 - e
    w = d / d.sum()
    return pd.Series(w, index=cols), pd.Series(e, index=cols), pd.Series(d, index=cols), Z, (Z * w).sum(axis=1)
Robustness is tested with equal weights, two alternative normalisations, leave-one-country-out, 5,000 perturbations of the indicator values and codes, and all 637 country subsets of five to nine countries with the weights recomputed. The ranking is sensitive to the composition of the selected country set.

2.5 Sensitivity Analysis
PRCC is computed on the residuals of the ranked input and the ranked output after regression on the ten other ranked inputs. The p-value uses a t distribution with N-2-10 degrees of freedom. The confidence interval uses the Fisher transformation with the standard error 1/sqrt(N-3-10). PRCC is read as conditional monotonic association.

def prcc(S, out):
    names = list(PR); X = np.column_stack([rankdata(S[PR[n]]) for n in names]); Y = rankdata(out)
    for k, n in enumerate(names):
        Zc = np.column_stack([np.delete(X, k, axis=1), np.ones(len(Y))])
        rx = X[:, k] - Zc @ np.linalg.lstsq(Zc, X[:, k], rcond=None)[0]
        ry = Y - Zc @ np.linalg.lstsq(Zc, Y, rcond=None)[0]
        r = pearsonr(rx, ry)[0]
        df_ = len(Y) - 2 - (len(names) - 1)
        tt = r * np.sqrt(df_ / (1 - r ** 2)); p = 2 * t_dist.sf(abs(tt), df_)
        se = 1 / np.sqrt(len(Y) - 3 - (len(names) - 1))
        lo, hi = np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)
The contribution of each input to the variance of LCOP is estimated with Sobol first-order and total-order indices (Saltelli and Jansen estimators, 200,000 base samples, 300 bootstrap resamples).

yA = model(draw(A_))[1]; yB = model(draw(B_))[1]
for name, key in PR.items():
    AB = A_.copy(); AB[:, col[key]] = B_[:, col[key]]
    yAB = model(draw(AB))[1]
    var = np.var(np.concatenate([yA, yB]))
    S1 = np.mean(yB * (yAB - yA)) / var
    ST = 0.5 * np.mean((yA - yAB) ** 2) / var
The one-at-a-time scenarios are conditional results of this cost model. They are not an assessment of policy instruments. A plant-scale sensitivity multiplies the synthesis cost component only (factors 1.30 and 1.50) on the same samples. The factor 1.30 is motivated by the reported cost increase for plants below about 30 kt/a. The factor 1.50 is a stress test. Neither is a literature-derived scale coefficient.

for n_, f_ in [("Baseline", 1.0), ("Scale-adjusted", 1.30), ("High scale penalty", 1.50)]:
    S_ = draw(U); S_["synthesis_cost_usd_per_kg"] = S_["synthesis_cost_usd_per_kg"] * f_
    _, p_, _, _ = model(S_)
2.6 Time-Horizon Projection
The projection for 2030 and 2050 applies scenario-based percentage changes relative to 2025 to CAPEX, CO2 price, electricity price and synthesis cost, and a horizon-specific range to the specific electricity consumption. These are not experience curves based on cumulative deployment. The base inputs, including the stoichiometric ratios, are drawn once and retained along all horizons. The quantile of each change variable is the same in 2030 and 2050.

U_t = rng_t.random((N_RM, 4))
S["capex_usd_per_kw"] = Sb["capex_usd_per_kw"] * (1 - tri_ppf(U_t[:, 0], *rate("capex_usd_per_kw", year)))
S["co2_price_usd_per_t"] = Sb["co2_price_usd_per_t"] * (1 - tri_ppf(U_t[:, 1], *rate("co2_price_usd_per_t", year)))
S["elec_price_usd_per_mwh"] = Sb["elec_price_usd_per_mwh"] * (1 + tri_ppf(U_t[:, 2], *rate("elec_price_usd_per_mwh", year)))
S["synthesis_cost_usd_per_kg"] = Sb["synthesis_cost_usd_per_kg"] * (1 - tri_ppf(U_t[:, 3], *rate("synthesis_cost_usd_per_kg", year)))
Parameters
Parameter	Unit	Minimum / mode / maximum
CAPEX	USD/kW	900 / 1500 / 2500
Specific electricity consumption	kWh/kg H2	46 / 51 / 57
Capacity factor	-	0.3 / 0.55 / 0.9
WACC	-	0.05 / 0.07 / 0.1
Equipment lifetime	years	20 / 25 / 30
Fixed O&M rate	per year	0.02 / 0.025 / 0.04
Electricity price (aggregate model)	USD/MWh	40 / 70 / 150
Hydrogen ratio (alpha)	kg H2/kg product	0.44 / 0.5 / 0.62
CO2 ratio (beta)	kg CO2/kg product	3.1 / 3.16 / 3.9
CO2 price	USD/t	150 / 350 / 700
Synthesis cost (gamma)	USD/kg product	0.8 / 1.3 / 2.2
Symbol	Meaning	Value
N	Monte Carlo draws	20,000
N_sobol	Sobol base samples	200,000
N_boot	Bootstrap resamples (Sobol)	300
N_perturb	Perturbations of the readiness index	5,000
FX	USD per EUR	1.09
s_0	Random seed	42
Data Sources
Electricity Prices
Monthly means of the 2025 day-ahead wholesale prices by country (Ember, 2026, based on ENTSO-E data). The prices are not the prices paid by an electrolyser.

Projects
Synfuel projects of the IEA Hydrogen Production and Infrastructure Projects Database (June 2026 release), used for criterion C4. The project-level records of the ten countries are in the sheet C4_Projects.

Input Workbook (data.xlsx)
Sheet	Content
MC_Parameters	Minimum, mode and maximum of each input with source and basis
Country_Electricity, Country_Elec_Monthly_2025	Country selection and the twelve monthly mean prices
EWM_Indicators, Country_Grid_Mix	Raw indicator values for C2, C4, C5, C6 and C7
C4_Project_Status, C4_Projects	Project counts by status and project-level records behind C4
EWM_Coding_Rules	Definition, direction and coding rule of each criterion
Roadmap_LearningRates	Scenario changes for 2030 and 2050
Reference_Case_Wolf2024	Reference case used to check Equation 2
Settings	Sample sizes, seed and exchange rate
Sources	Data sources and links
Software Environment
Package	Version	Role
Python	3.13	Core implementation
NumPy	2.5	Array computation and random-number generation
SciPy	1.18	Rank transformation, correlation and t distribution
pandas	3.0	Data handling
Matplotlib	3.11	Figure generation
openpyxl	3.1	Workbook input and output
How to Read the Results
Headline results of the paper with the seed above: median LCOH 6.33 USD/kg, median LCOP 6.09 USD/kg, country medians of LCOP from 5.00 USD/kg (Sweden) to 7.39 USD/kg (Italy), and a median LCOP of 4.01 USD/kg in 2050 under the stated scenarios. The results are conditional on the input ranges, which are scenario ranges and not forecasts. Prices are given at the price level of the source publications without inflation harmonisation. The excerpts in this repository illustrate the method.

References
[1] Wolf et al. (2024) reference case for the 2050 LCOH of EU countries, as cited in the article.
[2] Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S. Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. Comput Phys Commun 2010;181:259-70.
[3] Jansen MJW. Analysis of variance designs for model output. Comput Phys Commun 1999;117:35-43.
[4] Blower SM, Dowlatabadi H. Sensitivity and uncertainty analysis of complex models of disease transmission: an HIV model, as an example. Int Stat Rev 1994;62:229-43.
Citation
If you use this code or methodology in your research, please cite:

Mert, İ., & Yağlı, H. (2026). Hydrogen Cost Sensitivity in the e-SAF Economy: A Stochastic LCOH-LCOP Linkage Model for Net-Zero Aviation in the EU. Energy Conversion and Management (under review).
License
This code is made available under the MIT License.

Contact and Data Availability
The numerical input data of the study are provided in data.xlsx. The complete analysis script is not publicly available because it forms part of ongoing research. Summary results of all analyses are provided in the Supplementary Material of the paper.

Repository maintained by: İlker Mert (ilkermert@osmaniye.edu.tr) For questions regarding the methodology, please contact: İlker Mert, ilkermert@osmaniye.edu.tr

This README accompanies representative code excerpts for the article referenced above. Last updated: November 2026
