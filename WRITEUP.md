# Merit Order Effect in German Electricity Markets (2018–2026)
### Quantifying Renewable Energy's Price-Suppressing Impact on EPEX Day-Ahead Prices

**Author:** Soham Ajay Deo | B.E. EEE + M.Sc. Economics, BITS Pilani Goa  
**Data Source:** [SMARD](https://www.smard.de/) | **Period:** October 2018 – March 2026

---

## 1. Research Question

Does higher renewable generation (wind + solar) statistically suppress German day-ahead electricity prices via the merit order effect, and has this relationship changed structurally following the 2021–2022 European energy crisis?

Secondary question: Can renewable penetration and temporal features predict whether a given hour will have a negative electricity price?

---

## 2. Data

### 2.1 Source and Series

All data is sourced from the SMARD API operated by Germany's federal network agency. The API is public and provides hourly resolution data for the German electricity market. Data was fetched using API.

| Series | SMARD Filter ID | Unit |
|---|---|---|
| Day-ahead price (DE-LU bidding zone) | 4169 | €/MWh |
| Grid load | 410 | MW |
| Solar PV generation | 4068 | MW |
| Wind offshore generation | 1225 | MW |
| Wind onshore generation | 4067 | MW |

### 2.2 Why These Series

**Day-ahead price:** The EPEX SPOT day-ahead auction price for the DE-LU bidding zone is the standard market signal studied for merit order data analysis. It is the price at which the next day's electricity is contracted, determined by a uniform-price auction where generators submit supply bids and the market clears at the intersection with forecasted demand. This is the dependent variable throughout.

**Grid load:** Included as a measure of demand. Load determines where on the supply curve the market clears — high load means more expensive dispatchable plants are needed, raising the marginal price. This is a necessary control variable.

**Solar, wind onshore, wind offshore separately:** These three series are fetched individually rather than as an aggregate because they differ in their behavior — solar peaks midday in summer, onshore wind peaks in winter, offshore wind is more stable. Fetching them separately allows both aggregation (for regression) and individual use (for the ML feature matrix where the model can find interactions between them).

### 2.3 Start Date Decision

`START_MS = 1538344800000` corresponds to **30 September 2018 22:00 UTC** — the earliest timestamp for which day-ahead price data is available. The generation series have data from December 2014, but since DAP is the dependent variable the analysis window is constrained by DAP availability. Starting earlier would have no DAP to regress against.

### 2.4 Merging Decision: Inner Join

The five series are merged on timestamp using an inner join — keeping only timestamps present in all five series simultaneously.

**Why inner join and not outer:** An outer join would introduce NaN values wherever any single series had a gap. Since the regression requires complete observations across all variables, NaN rows would either need to be dropped anyway or imputed. Inner join is honest — it only keeps hours where all five measurements exist, making the data quality assumption explicit rather than hidden.

**Row loss:** 65,473 DAP rows → 65,438 after inner join. Loss of 35 rows. This is entirely explained by the trailing edge: DAP data extends to March 20 2026 while generation series end around March 19 2026, due to SMARD's ~24-hour publication lag for actual generation data. There are no internal gaps.

### 2.5 Descriptive Statistics

| Variable | Mean | Std | Min | Max |
|---|---|---|---|---|
| DAP (€/MWh) | 93.56 | 91.47 | −500.00 | 936.28 |
| Grid Load (MW) | 55,029 | 9,732 | 30,903 | 81,320 |
| Solar (MW) | 6,035 | 9,591 | 0 | 52,132 |
| Wind Offshore (MW) | 2,910 | 1,945 | 0 | 8,442 |
| Wind Onshore (MW) | 12,062 | 9,486 | 47 | 48,500 |
| Renewable Penetration | 0.382 | 0.226 | 0.002 | 1.203 |

Renewable penetration exceeds 1.0 in some hours (maximum 1.203), meaning renewables briefly exceeded grid load. This occurs during export hours when Germany's surplus flows to neighbouring grids and is not a data error.

**Negative price hours:** 2,100 out of 65,438 (3.21%), heavily concentrated in post-2023.

---

## 3. Variable Construction Decisions

### 3.1 total_renewables

$$\text{total\\_renewables}_t = \text{solar}_t + \text{wind\\_onshore}_t + \text{wind\\_offshore}_t$$

Simple sum in MW. Used in EDA scatter plots to visualise the raw merit order relationship. Not used in the final regression — see Section 3.2.

### 3.2 renewable_penetration — Why This Instead of total_renewables

$$\text{renewable\\_penetration}_t = \frac{\text{total\\_renewables}_t}{\text{grid\\_load}_t}$$

**The problem with simply summing up:** Germany's renewable capacity has grown substantially over 2018–2026. 20,000 MW of renewables in 2019 represents a very different market condition than 20,000 MW in 2024 — the grid has more capacity, storage has expanded, and interconnection has changed. Using raw MW conflates capacity growth over time with within-period merit order dynamics. The coefficient would be unstable across time.

**Why the ratio is better:** Penetration normalises renewable output to contemporaneous demand. A penetration of 0.8 means renewables are covering 80% of load at that moment — this is the economically meaningful quantity for the merit order effect, because it directly measures how far down the merit order thermal plants are being pushed.

**The multicollinearity confirmation:** The correlation matrix confirmed this decision quantitatively:

| | total_renewables | gl | renewable_penetration |
|---|---|---|---|
| total_renewables | 1.000 | 0.275 | 0.948 |
| gl | 0.275 | 1.000 | 0.000 |
| renewable_penetration | 0.948 | 0.000 | 1.000 |

`renewable_penetration` has correlation 0.948 with `total_renewables` — they contain almost identical information. But `renewable_penetration` has correlation 0.000 with `gl` — they are orthogonal. Using penetration rather than raw MW plus load separately eliminates the multicollinearity problem, as confirmed by the condition number dropping from $1.54 \times 10^6$ to 29.4.

### 3.3 is_negative

$$\text{is\\_negative}_t = \mathbf{1}[\text{DAP}_t < 0]$$

Created as integer (0/1) rather than boolean because statsmodels' logit requires numeric input — boolean dtype causes a `ValueError`.

### 3.4 Crisis Period Definition

$$\text{crisis\\_dummy}_t = \mathbf{1}[2021\text{-}06\text{-}01 \leq t < 2023\text{-}01\text{-}01]$$

**Why June 2021 as start, not February 2022 (Russia-Ukraine invasion):**

The invasion is commonly cited as the cause of the European energy crisis, but electricity prices began rising sharply in mid-2021 — over six months earlier. The causes were: post-COVID gas demand recovery, unusually low European gas storage entering summer 2021, and reduced Russian pipeline flows beginning in H2 2021. By Q4 2021, German day-ahead prices were already 2–3× their pre-crisis baseline. Using March 2022 as the start would misclassify Q3–Q4 2021 as pre-crisis, contaminating the pre-crisis baseline and biasing the period-split regression coefficients upward. The DAP time series confirms price elevation is clearly visible from mid-2021.

**Why January 2023 as end:** European gas prices peaked August 2022 and declined sharply through H2 2022 as LNG import infrastructure came online and demand destruction took effect. By January 2023 prices had largely normalised to a structurally stable post-crisis level.

---

## 4. Regression Methodology

### 4.1 The Estimating Equation

$$\text{DAP}_t = \beta_0 + \beta_1 \cdot \text{pen}_t + \sum_{h=1}^{23} \gamma_h \cdot \mathbf{1}[\text{hour}_t = h] + \sum_{m=2}^{12} \delta_m \cdot \mathbf{1}[\text{month}_t = m] + \sum_{d=1}^{6} \lambda_d \cdot \mathbf{1}[\text{dow}_t = d] + \phi \cdot \text{crisis}_t + \varepsilon_t$$

### 4.2 Why Each Control Variable Was Included

**Hour of day — categorical not continuous:**

Hour is included as $C(\text{hour})$ — 23 dummies for hours 1–23 with hour 0 (midnight) as reference. If hour were continuous, the model would assume each additional hour adds the same fixed price increment — a linear relationship. This is false. Electricity prices follow a non-linear double-peaked daily profile: low overnight, a morning ramp, a midday solar dip, an evening peak, then decline. There is no linear structure. The categorical specification lets each hour have its own independent baseline without assuming any specific form. The regression confirms this: hour 17 has coefficient +64.13 €/MWh while hours 1–3 are near zero. A linear variable would fit a straight line through a wave pattern.

**Month — seasonal demand and generation patterns:**

Two mechanisms make month important. First, demand is seasonal — January heating loads are higher than May. Second, renewable generation is seasonal — solar peaks June–August, wind peaks November–February. Both correlate with price. Without month controls, the penetration coefficient partially absorbs seasonal variation rather than isolating the within-season merit order effect.

**Day of week — weekend demand collapse:**

Industrial load drops 15–25% on weekends regardless of season, weather, or renewable output. Without this control, the model incorrectly attributes lower weekend prices to higher renewable penetration, since weekends also tend to have higher penetration due to lower denominator load. Negative prices cluster heavily on weekends — day-of-week is particularly important for the logistic regression.

**Crisis dummy — structural break isolation:**

The 2022 crisis shifted the price level up ~€122/MWh through a mechanism completely unrelated to renewables — gas supply disruption raising the marginal cost of every dispatchable plant. Without this control, the model sees high prices coexisting with medium renewable output during 2021–2022 and partially attributes those high prices to low renewable penetration, attenuating the merit order effect estimate toward zero.

### 4.3 Why Grid Load Was Dropped

An initial specification included both `renewable_penetration` and `gl`. The result was:
- Condition number: $1.54 \times 10^6$ — severe multicollinearity
- `gl` coefficient: −0.0013 (p < 0.001) — **economically wrong sign**

Higher load should increase price. A negative coefficient means the model says higher demand lowers price — nonsensical. This sign reversal is a classic multicollinearity symptom: when two predictors are correlated, their coefficients become unstable and can flip sign. The correlation matrix confirmed the cause ($\rho = 0.948$ between penetration and total_renewables, with `gl` embedded in the penetration denominator).

Dropping `gl` reduced the condition number from $1.54 \times 10^6$ to 29.4 at a cost of only 0.4 percentage points of $R^2$ (0.462 → 0.458). The hour and month dummies already absorb the legitimate load variation. This is the correct specification.

### 4.4 Why HAC Standard Errors with 24 Lags

OLS assumes $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ for $t \neq s$. This is violated in hourly electricity data — the price at 3pm depends on conditions that also affect 4pm. The Durbin-Watson statistic of 0.053 (benchmark: 2.0 for no autocorrelation) confirms severe serial correlation.

When violated, OLS coefficients remain unbiased but standard errors are underestimated — 65,000 correlated observations are treated as 65,000 independent draws, massively overstating effective sample size. The practical consequence: under naive OLS, `gl` had t-statistic −6.03 (p < 0.001). Under HAC, it dropped to −1.85 (p = 0.065) — no longer significant. The naive model was lying about certainty. The key renewable penetration coefficient standard error widened approximately 4× under HAC. 24 lags covers one full daily cycle — the most significant autocorrelation structure in hourly electricity prices is intraday. 24 captures this without over-correcting for longer-horizon dependencies.

### 4.5 Why Period Splits Rather Than Just the Crisis Dummy

The crisis dummy shifts the **intercept** — it adds a constant to the price level during the crisis. It does not allow the **slope** on renewable penetration to change between periods.

The research question is whether the merit order effect itself has changed — whether renewables are now more powerful at suppressing prices than before. This requires comparing the $\beta_1$ coefficient across periods, not just the intercept. Period splits fit entirely separate models on pre-crisis and post-crisis subsamples, letting every coefficient vary freely. The finding — $\hat{\beta}_{pre} = -65.78$ versus $\hat{\beta}_{post} = -171.91$ — would have been invisible with just a crisis dummy.

---

## 5. Logistic Regression Methodology

### 5.1 Why Logistic and Not OLS for is_negative

`is_negative` is binary (0 or 1). OLS on a binary outcome produces:
1. **Unbounded predictions** above 1 or below 0 — not valid probabilities
2. **Linear probability assumption** — constant marginal effect across the full range, when the actual penetration-to-negative-price relationship is clearly S-shaped
3. **Heteroscedasticity by construction** — $\text{Var}(Y) = p(1-p)$ depends on the predicted probability

Logistic regression handles all three:

$$P(Y_t = 1 \mid X_t) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_t + \ldots)}}$$

The logistic function maps the linear combination to $(0,1)$, respects bounded probability, and captures the nonlinear S-shaped relationship naturally.

### 5.2 Why Marginal Effects Are Reported, Not Raw Coefficients

Raw logistic coefficients are in log-odds units — a coefficient of 15.52 means a one-unit increase in penetration increases the log-odds by 15.52. This is not directly interpretable. Marginal effects convert to probability units:

$$\frac{\partial P}{\partial x_j} = \beta_j \cdot P(1-P)$$

evaluated at the sample mean of $P$. The result is the percentage point change in probability per unit change in the predictor, averaged over the dataset.

### 5.3 Why crisis_dummy Was Excluded from Logistic Regression

Negative price hours are almost exclusively a post-crisis phenomenon. Including crisis_dummy in the logistic model mixes two mechanisms: the probability of a negative price conditional on penetration versus the unconditional change in that probability across regimes. The logistic model is intended as a predictive tool describing the penetration-to-negative-price relationship — fitting without the dummy is appropriate for this purpose.

---

## 6. Machine Learning Methodology

### 6.1 Why Gradient Boosting

**vs Random Forest:** Random forest builds trees independently in parallel; gradient boosting builds trees sequentially where each tree corrects the errors of the ensemble so far. For structured tabular data with threshold effects (price behaviour changes sharply around penetration ≈ 0.8–1.0), gradient boosting's sequential correction is more effective at capturing discontinuities.

**vs Neural Network:** Requires extensive hyperparameter tuning, longer training time, and produces no interpretable feature importances. The feature importance output — renewable_penetration at 67% — connects directly back to the OLS results and is a substantive finding. A neural network cannot provide this connection cleanly.

**vs Logistic Regression:** Cannot capture variable interactions without manual feature engineering. Gradient boosting discovers interactions automatically through tree splitting — e.g. the combination of high solar + Sunday + overnight that produces negative prices, without being explicitly specified.

### 6.2 Why Time-Based Split

Random train/test splitting is invalid for time series data — future observations leak into training, allowing the model to learn from information unavailable in real deployment. The model would be evaluated on "predicting the past given the future" — trivially easier than genuine forecasting and producing inflated accuracy.

**Training:** October 2018 – December 2024 (54,818 hours)  
**Test:** January 2025 – March 2026 (10,620 hours)

The model learns from 6 years and is evaluated on 15 months it has never seen.

### 6.3 Why F1 Score

Test set contains 598 negative hours out of 10,620 total — 5.6% base rate. A trivial classifier predicting "never negative" achieves 94.4% accuracy while being completely useless.

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

F1 penalises both false positives (false alarms) and false negatives (missed events). It is the correct metric for imbalanced binary classification.

### 6.4 Why Individual Generation Sources in Feature Matrix

The logistic regression uses only `renewable_penetration`. The gradient boosting feature matrix includes `slr`, `won`, `wof` individually alongside penetration. Solar-dominated oversupply (high slr, low wind) is structurally different from wind-dominated oversupply even at the same penetration ratio — solar is highly intermittent and peaky, wind oversupply tends to be more sustained. Including all three gives the model information to discover these differences if they exist, without pre-specifying the interaction.

---

## 7. Results

### 7.1 OLS Results

**Table 1: OLS Coefficient on Renewable Penetration Across Specifications**

| Model | Specification | $\hat{\beta}_{\text{pen}}$ | HAC Std Err | $R^2$ | Cond. No. |
|---|---|---|---|---|---|
| Naive | pen only | −0.0023/MW | — | 0.103 | — |
| Base (HAC) | pen + controls + crisis | −128.29 | 4.672 | 0.458 | 29.4 |
| Pre-crisis | pen + controls (HAC) | −65.78 | 1.857 | 0.606 | — |
| Post-crisis | pen + controls (HAC) | −171.91 | 5.263 | 0.598 | — |

$\hat{\beta}_{\text{pen}} = -128.29$ (p < 0.001, 95% CI: [−137.45, −119.13]). A 0.1 unit increase in penetration suppresses price by **€12.83/MWh**. Crisis dummy $\hat{\phi} = 122.08$ — gas supply shock added €122/MWh independent of renewables.

The period split is the key finding:

$$\Delta\hat{\beta}_{\text{pen}} = -171.91 - (-65.78) = -106.13 \quad (+161\%)$$

### 7.2 Logistic Regression

$$\frac{\partial P(\text{DAP} < 0)}{\partial \text{pen}} = +0.2454 \quad \text{(p < 0.001)}$$

Each 0.1 unit increase in penetration raises negative price probability by **2.5 percentage points**. Peak risk: Sunday, May, overnight.

### 7.3 Gradient Boosting

| Model | Precision | Recall | F1 (Class 1) |
|---|---|---|---|
| Logistic Regression | 0.69 | 0.71 | 0.70 |
| Gradient Boosting | 0.72 | 0.77 | **0.74** |

77% of negative price hours correctly identified on unseen 2025–2026 data. Renewable penetration: 67% of total feature importance.

---

## 8. Conclusions

**Finding 1:** Renewable penetration suppresses German day-ahead prices by €128.29/MWh per unit penetration, controlling for load patterns, seasonality, and the energy crisis. Robust to HAC correction.

**Finding 2:** The merit order effect strengthened 161% post-crisis (−65.78 → −171.91). Germany's Energiewende acceleration post-2022 is producing measurably larger price suppression as the fleet expands.

**Finding 3:** 3.21% of hours had negative prices, concentrated post-2023. Gradient boosting predicts 77% of negative price hours out-of-sample, with renewable penetration at 67% feature importance.

---

## 9. Limitations

- **Residual autocorrelation (DW = 0.053):** HAC corrects standard errors but does not eliminate serial correlation. A time-series model (ARIMA-X, VAR) would handle dynamics more rigorously.
- **Missing variables:** Fuel prices, cross-border flows, and storage dispatch are excluded — these account for the 54% unexplained variance.
- **Quasi-separation in logistic model:** 48% of observations are perfectly predictable, inflating Pseudo $R^2$.
- **Crisis boundary involves judgement:** Results are robust to ±2 month variations.
- **No causal identification:** Estimates are consistent under conditional independence but do not claim strict causal identification.

---

## 10. Reproducibility

```bash
git clone https://github.com/sohamdeo05/renewable-price-suppression
cd renewable-price-suppression
pip install -r requirements.txt
jupyter notebook notebooks/final.ipynb
```

**Dependencies:** `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `matplotlib`, `requests`, `openpyxl`, `pyarrow`, `scikit-learn`
