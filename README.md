# Hydrogen Cost Sensitivity in the e-SAF Economy

**A Stochastic LCOH-LCOP Linkage Model for Net-Zero Aviation in the EU**

Authors: İlker Mert\*, Hüseyin Yağlı

This repository contains methodological documentation, representative code excerpts and the numerical input data for the research article (under review, *Energy Conversion and Management*).

## About the Study

The study propagates the uncertainty of the levelized cost of hydrogen (LCOH) to the levelized cost of production (LCOP) of e-SAF with a Monte Carlo simulation of 20,000 draws. It is applied to an aggregate model and to ten EU countries, and country readiness is scored with the Entropy Weight Method (EWM).

LCOP is the cost per kilogram of total hydrocarbon product of a CO2-to-hydrocarbon pathway. No cost is allocated to the jet fraction or to coproducts. The projection to 2050 is scenario-based. The country ranking is conditional on the selected country set.

| Stage | Content |
|---|---|
| core | Sampling, LCOH and LCOP, cost shares, correlation |
| country | Country-level LCOH and LCOP with national electricity price distributions |
| ewm | Readiness index, robustness to weights, normalisation, country subsets and perturbed indicators |
| sensitivity | PRCC, Sobol indices, scenarios, dependence scenarios, plant-scale sensitivity |
| horizon | Scenario-based projection for 2030 and 2050 |
| reference | Reproduction of an independent reference case (Wolf et al. [9]) |

## 1. Input Distributions and Sampling

All inputs are sampled from triangular distributions (minimum, mode, maximum), given in the Parameters table below. One matrix of uniform random numbers is drawn once and reused for every comparison between countries or horizons (common random numbers), so differences do not come from sampling noise. Inputs are independent in the baseline. Dependence is examined in separate scenarios.

The input ranges follow the literature-based Monte Carlo approach of Wolf et al. [9] and are documented in Table 1 of the article. The inverse CDF of the triangular distribution with minimum $a$, mode $m$ and maximum $b$, and $c=(m-a)/(b-a)$:

```math
x(u)=a+\sqrt{u\,(b-a)(m-a)}\quad\text{for } u<c
```

```math
x(u)=b-\sqrt{(1-u)(b-a)(b-m)}\quad\text{for } u\ge c
```

```python
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
```

## 2. LCOH and LCOP

```math
\mathrm{CRF}=\frac{w\,(1+w)^{L}}{(1+w)^{L}-1}
```

```math
\mathrm{LCOH}=\frac{\mathrm{CAPEX}\,(\mathrm{CRF}+f_{OM})\,\eta}{8760\cdot \mathrm{CF}}+\frac{P_e\,\eta}{1000}
```

```math
\mathrm{LCOP}=\alpha\,\mathrm{LCOH}+\beta\,\frac{P_{CO_2}}{1000}+\gamma
```

The LCOH equation and the capital recovery factor follow Wolf et al. [9]. CAPEX is in USD/kW, $\eta$ is the specific electricity consumption in kWh/kg H2 and $P_e$ is the electricity price in USD/MWh. The coefficients $\alpha$ and $\beta$ are effective feedstock requirements per kg of total hydrocarbon product. Their lower bounds are the stoichiometric minima of a jet-range n-alkane (C12H26), rounded to two decimals (0.44 and 3.10). The transfer function follows the stoichiometric conversion of [5]. The synthesis cost $\gamma$ excludes hydrogen and CO2 supply. Its mode is calibrated on the non-feedstock cost share reported by Bube et al. [8].

```python
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
```

## 3. Country-Level Application

For each country, the aggregate electricity price distribution is replaced by a triangular distribution defined by the minimum, median and maximum of the twelve monthly means of the 2025 day-ahead wholesale price (EUR/MWh, converted at 1.09 USD/EUR) from [33]. The prices are wholesale prices and exclude grid charges, taxes and procurement arrangements. All other draws are shared between the countries.

```python
for _, r in country_elec.iterrows():
    a, m, b = (r["elec_price_min_eur_mwh"] * FX, r["elec_price_mode_eur_mwh"] * FX, r["elec_price_max_eur_mwh"] * FX)
    Sc = draw(U, {"elec_price_usd_per_mwh": tri_ppf(U[:, col["elec_price_usd_per_mwh"]], a, m, b)})
    h, p, _, _ = model(Sc)
```

## 4. Readiness Index (EWM)

The index uses six criteria: C1 (LCOP), C2 (grid carbon intensity), C4 (maturity-weighted PtL project score), C5 (policy code 0 to 2), C6 (GDP per capita in PPS) and C7 (renewable generation over demand). The legal criterion C3 is not included. The index approach follows Neofytou et al. [24]. Grid intensity and the renewable-to-demand ratio are from Ember [56].

The C4 score of a country is the sum over its projects of the status weight times $(1+\ln(1+\text{capacity in kt H}_2\text{/y}))$.

The method follows Zhu et al. [34]. EWM has been applied to hydrogen systems [36], distributed energy systems [37] and renewable energy under uncertainty [38], and its sensitivity to normalisation is discussed in [39]. With normalised values $z_{ij}$, shares $p_{ij}=z_{ij}/\sum_i z_{ij}$, the entropy and weight of criterion $j$ are

```math
e_j=-\frac{1}{\ln n}\sum_{i} p_{ij}\ln p_{ij},\qquad w_j=\frac{1-e_j}{\sum_k (1-e_k)}
```

Terms with $p_{ij}=0$ are omitted ($0\ln 0=0$). The weights describe the variation of a criterion in the selected dataset. They do not establish its causal importance.

```python
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
```

Robustness is tested with equal weights, two alternative normalisations, leave-one-country-out, 5,000 perturbations of indicator values and codes, and all 637 country subsets of five to nine countries with the weights recomputed. The ranking is sensitive to the composition of the selected country set.

## 5. Sensitivity Analysis

PRCC is computed on the residuals of the ranked input and ranked output after regression on the ten other ranked inputs. The p-value uses a t distribution with $N-2-10$ degrees of freedom. The confidence interval uses the Fisher transformation with standard error $1/\sqrt{N-3-10}$. PRCC follows Marino et al. [25] and is read as conditional monotonic association.

```python
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
```

The contribution of each input to the variance of LCOP is estimated with Sobol first-order and total-order indices (Saltelli and Jansen estimators as described in the article, 200,000 base samples, 300 bootstrap resamples):

```math
S_i=\frac{\mathrm{E}\left[y_B\,(y_{AB_i}-y_A)\right]}{\mathrm{Var}(y)},\qquad S_{T_i}=\frac{\mathrm{E}\left[(y_A-y_{AB_i})^2\right]}{2\,\mathrm{Var}(y)}
```

```python
yA = model(draw(A_))[1]; yB = model(draw(B_))[1]
for name, key in PR.items():
    AB = A_.copy(); AB[:, col[key]] = B_[:, col[key]]
    yAB = model(draw(AB))[1]
    var = np.var(np.concatenate([yA, yB]))
    S1 = np.mean(yB * (yAB - yA)) / var
    ST = 0.5 * np.mean((yA - yAB) ** 2) / var
```

The one-at-a-time scenarios are conditional results of this cost model, not an assessment of policy instruments. The plant-scale sensitivity multiplies the synthesis cost component only (factors 1.30 and 1.50) on the same samples. The factor 1.30 is motivated by the cost increase above 30% reported in [8] for plants below about 30 kt/a. The factor 1.50 is a stress test. Neither is a literature-derived scale coefficient.

```python
for n_, f_ in [("Baseline", 1.0), ("Scale-adjusted", 1.30), ("High scale penalty", 1.50)]:
    S_ = draw(U); S_["synthesis_cost_usd_per_kg"] = S_["synthesis_cost_usd_per_kg"] * f_
    _, p_, _, _ = model(S_)
```

## 6. Time-Horizon Projection

The projection for 2030 and 2050 applies scenario-based percentage changes relative to 2025 to CAPEX, CO2 price, electricity price and synthesis cost, and a horizon-specific range to the specific electricity consumption. The changes are informed by learning-rate literature [41, 42, 43, 44] and electricity outlooks [45, 30], and are not experience curves based on cumulative deployment. The base inputs, including the stoichiometric ratios, are drawn once and kept along all horizons. The quantile of each change variable is the same in 2030 and 2050.

```python
U_t = rng_t.random((N_RM, 4))
S["capex_usd_per_kw"] = Sb["capex_usd_per_kw"] * (1 - tri_ppf(U_t[:, 0], *rate("capex_usd_per_kw", year)))
S["co2_price_usd_per_t"] = Sb["co2_price_usd_per_t"] * (1 - tri_ppf(U_t[:, 1], *rate("co2_price_usd_per_t", year)))
S["elec_price_usd_per_mwh"] = Sb["elec_price_usd_per_mwh"] * (1 + tri_ppf(U_t[:, 2], *rate("elec_price_usd_per_mwh", year)))
S["synthesis_cost_usd_per_kg"] = Sb["synthesis_cost_usd_per_kg"] * (1 - tri_ppf(U_t[:, 3], *rate("synthesis_cost_usd_per_kg", year)))
```

## Parameters

| Parameter | Unit | Minimum / mode / maximum |
|---|---|---|
| CAPEX | USD/kW | 900 / 1500 / 2500 |
| Specific electricity consumption | kWh/kg H2 | 46 / 51 / 57 |
| Capacity factor | - | 0.3 / 0.55 / 0.9 |
| WACC | - | 0.05 / 0.07 / 0.1 |
| Equipment lifetime | years | 20 / 25 / 30 |
| Fixed O&M rate | per year | 0.02 / 0.025 / 0.04 |
| Electricity price (aggregate model) | USD/MWh | 40 / 70 / 150 |
| Hydrogen ratio (alpha) | kg H2/kg product | 0.44 / 0.5 / 0.62 |
| CO2 ratio (beta) | kg CO2/kg product | 3.1 / 3.16 / 3.9 |
| CO2 price | USD/t | 150 / 350 / 700 |
| Synthesis cost (gamma) | USD/kg product | 0.8 / 1.3 / 2.2 |

| Symbol | Meaning | Value |
|---|---|---|
| N | Monte Carlo draws | 20,000 |
| N_sobol | Sobol base samples | 200,000 |
| N_boot | Bootstrap resamples (Sobol) | 300 |
| N_perturb | Perturbations of the readiness index | 5,000 |
| FX | USD per EUR | 1.09 |
| s_0 | Random seed | 42 |

## Data Sources

- **Electricity prices:** monthly means of 2025 day-ahead wholesale prices by country (Ember, 2026, based on ENTSO-E data). These are not prices paid by an electrolyser.
- **Projects:** synfuel projects of the IEA Hydrogen Production and Infrastructure Projects Database (June 2026 release), used for criterion C4. Project counts of the ten countries are given in the Input Data section.

## Input Data

The numerical inputs of the study are shown below. The values are those used for the results of the article.

### Input distributions

See the Parameters table above (Table 1 of the article).

### Monthly mean day-ahead wholesale electricity prices, 2025 (EUR/MWh)

The minimum, median and maximum of the twelve monthly means are the minimum, mode and maximum of the national triangular price distribution. Source: Ember [33], based on ENTSO-E data. The prices exclude grid charges, taxes and procurement arrangements.

| Country | Min | Median (mode) | Max |
|---|---|---|---|
| Sweden | 19.83 | 42.22 | 61.45 |
| Spain | 16.99 | 71.38 | 108.26 |
| Poland | 81.52 | 101.53 | 133.81 |
| Germany | 64.14 | 86.02 | 129.35 |
| Netherlands | 64.21 | 85.64 | 125.81 |
| Czechia | 73.12 | 93.77 | 132.62 |
| France | 19.44 | 57.66 | 122.63 |
| Austria | 66.67 | 98.09 | 140.77 |
| Romania | 78.26 | 104.94 | 157.17 |
| Italy | 109.10 | 119.31 | 142.33 |

### Readiness index indicators

C1 (LCOP) is a model output. C3 is not included. C2, C5, C6 and C7 as defined in the article; C4 is the maturity-weighted score of PtL project records.

| Country | C2 grid intensity (gCO2/kWh) | C4 project score | C5 policy code | C6 GDP per capita (PPS, EU=100) | C7 renewables / demand (%) |
|---|---|---|---|---|---|
| Sweden | 35.3 | 36.55 | 0 | 115 | 88.9 |
| Spain | 153.6 | 99.32 | 1 | 90 | 58.5 |
| Poland | 590.9 | 0.00 | 0 | 85 | 31.1 |
| Germany | 330.0 | 57.60 | 2 | 125 | 56.8 |
| Netherlands | 253.6 | 30.46 | 1 | 134 | 57.2 |
| Czechia | 401.9 | 0.00 | 0 | 91 | 18.5 |
| France | 41.5 | 92.78 | 1 | 95 | 31.2 |
| Austria | 117.2 | 0.00 | 1 | 125 | 79.1 |
| Romania | 259.6 | 0.00 | 0 | 72 | 43.7 |
| Italy | 284.7 | 10.45 | 0 | 93 | 41.4 |

### PtL project records behind C4

Projects of the IEA Hydrogen Production and Infrastructure Projects Database (June 2026 release), synfuel end-use flag.

| Country | Projects | Announced | Demo | FID/Construction | Operational | Other | Capacity (kt H2/y) |
|---|---|---|---|---|---|---|---|
| Sweden | 8 | 6.0 | 0 | 0 | 0 | 2.0 | 420.5 |
| Spain | 18 | 16.0 | 1 | 1 | 0 | 0.0 | 668.8 |
| Poland | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0.0 |
| Germany | 20 | 7.0 | 6 | 1 | 2 | 4.0 | 469.3 |
| Netherlands | 7 | 4.0 | 1 | 0 | 0 | 2.0 | 481.2 |
| Czechia | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0.0 |
| France | 21 | 13.0 | 1 | 0 | 0 | 7.0 | 912.1 |
| Austria | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0.0 |
| Romania | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0.0 |
| Italy | 2 | 2.0 | 0 | 0 | 0 | 0.0 | 34.7 |

### Scenario changes for 2030 and 2050 (relative to 2025)

Fractional reduction is applied as (1 - change) and fractional change as (1 + change). Efficiency is given as an absolute range in kWh/kg H2.

| Variable | Year | Type | Min | Mode | Max |
|---|---|---|---|---|---|
| capex_usd_per_kw | 2030 | fractional reduction | 0.15 | 0.35 | 0.5 |
| capex_usd_per_kw | 2050 | fractional reduction | 0.35 | 0.65 | 0.8 |
| co2_price_usd_per_t | 2030 | fractional reduction | 0.1 | 0.25 | 0.4 |
| co2_price_usd_per_t | 2050 | fractional reduction | 0.4 | 0.65 | 0.85 |
| elec_price_usd_per_mwh | 2030 | fractional change | -0.2 | 0 | 0.15 |
| elec_price_usd_per_mwh | 2050 | fractional change | -0.15 | 0.1 | 0.3 |
| synthesis_cost_usd_per_kg | 2030 | fractional reduction | 0.02 | 0.08 | 0.15 |
| synthesis_cost_usd_per_kg | 2050 | fractional reduction | 0.08 | 0.2 | 0.35 |
| efficiency_kwh_per_kg | 2025 | absolute kwh per kg | 46 | 51 | 57 |
| efficiency_kwh_per_kg | 2030 | absolute kwh per kg | 44 | 49 | 54 |
| efficiency_kwh_per_kg | 2050 | absolute kwh per kg | 38 | 43 | 48 |

## Software Environment

| Package | Version | Role |
|---|---|---|
| Python | 3.13 | Core implementation |
| NumPy | 2.5 | Array computation and random-number generation |
| SciPy | 1.18 | Rank transformation, correlation and t distribution |
| pandas | 3.0 | Data handling |
| Matplotlib | 3.11 | Figure generation |
| openpyxl | 3.1 | Workbook input and output |

## How to Read the Results

Headline results of the paper (seed 42): median LCOH 6.33 USD/kg, median LCOP 6.09 USD/kg, country median LCOP from 5.00 USD/kg (Sweden) to 7.39 USD/kg (Italy), and a median LCOP of 4.01 USD/kg in 2050 under the stated scenarios. Results are conditional on the input ranges, which are scenario ranges and not forecasts. Prices are at the price level of the source publications, without inflation harmonisation.

## References

Numbering follows the reference list of the article.

[5] Drünert, S., Neuling, U., Zitscher, T., & Kaltschmitt, M. (2020). Power-to-Liquid fuels for aviation: Processes, resources and supply potential under German conditions. *Applied Energy*, 277, 115578. https://doi.org/10.1016/j.apenergy.2020.115578

[8] Bube, S., Voß, S., Quante, G., & Kaltschmitt, M. (2025). Cost analysis of kerosene production from power-based syngas via the Fischer-Tropsch and methanol pathway. *Fuel*, 384, 133901. https://doi.org/10.1016/j.fuel.2024.133901

[9] Wolf, N., Tanneberger, M. A., & Höck, M. (2024). Levelized cost of hydrogen production in Northern Africa and Europe in 2050: A Monte Carlo simulation for Germany, Norway, Spain, Algeria, Morocco, and Egypt. *International Journal of Hydrogen Energy*, 69, 184–194. https://doi.org/10.1016/j.ijhydene.2024.04.319

[24] Neofytou, H., Nikas, A., & Doukas, H. (2020). Sustainable energy transition readiness: A multicriteria assessment index. *Renewable and Sustainable Energy Reviews*, 131, 109988. https://doi.org/10.1016/j.rser.2020.109988

[25] Marino, S., Hogue, I. B., Ray, C. J., & Kirschner, D. E. (2008). A methodology for performing global uncertainty and sensitivity analysis in systems biology. *Journal of Theoretical Biology*, 254(1), 178–196. https://doi.org/10.1016/j.jtbi.2008.04.011

[30] International Energy Agency. (2022). Direct air capture: A key technology for net zero. OECD Publishing. https://doi.org/10.1787/bbd20707-en

[33] Ember. (2026). European wholesale electricity price data, monthly (day-ahead prices from ENTSO-E) [Data set]. https://ember-energy.org/data/european-wholesale-electricity-price-data/

[34] Zhu, Y., Tian, D., & Yan, F. (2020). Effectiveness of entropy weight method in decision-making. *Mathematical Problems in Engineering*, 2020, 3564835. https://doi.org/10.1155/2020/3564835

[36] Li, Z., Wang, Y., Xie, J., Cheng, Y., & Shi, L. (2024). Hybrid multi-criteria decision-making evaluation of multiple renewable energy systems considering the hysteresis band principle. *International Journal of Hydrogen Energy*, 49, 450–462. https://doi.org/10.1016/j.ijhydene.2023.09.059

[37] Wang, W., Li, H., Hou, X., Zhang, Q., & Tian, S. (2021). Multi-criteria evaluation of distributed energy system based on order relation-anti-entropy weight method. *Energies*, 14(1), 246. https://doi.org/10.3390/en14010246

[38] Sitorus, F., & Brito-Parada, P. R. (2020). A multiple criteria decision making method to weight the sustainability criteria of renewable energy technologies under uncertainty. *Renewable and Sustainable Energy Reviews*, 127, 109891. https://doi.org/10.1016/j.rser.2020.109891

[39] Roszkowska, E., & Wachowicz, T. (2024). Impact of normalization on entropy-based weights in Hellwig's method: A case study on evaluating sustainable development in the education area. *Entropy*, 26(5), 365. https://doi.org/10.3390/e26050365

[41] Glenk, G., Meier, R., & Reichelstein, S. (2021). Cost dynamics of clean energy technologies. *Schmalenbach Journal of Business Research*, 73(2), 179–206. https://doi.org/10.1007/s41471-021-00114-8

[42] International Renewable Energy Agency. (2020). Green hydrogen cost reduction: Scaling up electrolysers to meet the 1.5°C climate goal. IRENA, Abu Dhabi. https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2020/Dec/IRENA_Green_hydrogen_cost_2020.pdf

[43] CleanTechnica. (2025, February 24). Hydrogen electrolysis cost projections from major organizations low by 60% to 300%. https://cleantechnica.com/2025/02/24/hydrogen-electrolysis-cost-projections-from-major-organizations-low-by-60-to-300/

[44] National Energy Technology Laboratory. (2013). Technology learning curve (FOAK to NOAK) (Report No. DOE/NETL-341/081213). U.S. Department of Energy.

[45] Montel Energy / Energy Brainpool. (2025). EU energy outlook: How will Europe evolve over the next 30 years? https://montel.energy/resources/blog/eu-energy-outlook-2050-how-will-europe-evolve-over-the-next-30-years

[56] Ember. (2026). Yearly electricity data [Data set]. https://ember-energy.org/data/yearly-electricity-data/

## Citation

If you use this code or methodology, please cite:

> Mert, İ., & Yağlı, H. (2026). Hydrogen Cost Sensitivity in the e-SAF Economy: A Stochastic LCOH-LCOP Linkage Model for Net-Zero Aviation in the EU. *Energy Conversion and Management* (under review).

## License

Released under the MIT License (see `LICENSE`).

## Contact and Data Availability

The numerical input data of the study are given in the tables of this README, and representative code excerpts of the core methodological steps are given in the method sections above. The complete analysis script is not shared at this time because the same code base is being used in other ongoing studies. It is available from the corresponding author on reasonable request. Summary results of all analyses are provided in the Supplementary Material of the paper.

Maintainer: İlker Mert (ilkermert@osmaniye.edu.tr)

> **Note on scope.** The code blocks above are representative excerpts of the core methodological steps, included for transparency. This repository does not allow all tables and figures of the paper to be reproduced by running a script.
