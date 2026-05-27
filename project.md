# Predicting House Sale Prices with Statistical Modeling and Neural Networks
**Project Report**

**Student 1:** Ali Mohamed Saleh [24502783]  
**Student 2:** Haddad Youssef Bilal [2383725]  

*Computational Statistics, Spring 2026, submitted on May 27, 2026*

---

## Abstract
Predicting house prices often feels like hitting a moving target: a home isn’t just a simple sum of rooms, but a complex mix of features and market psychology. This project dissects the Ames housing dataset by combining classical statistics with artificial intelligence. 

Right from the start, we faced a major hurdle: a handful of overpriced luxury homes were skewing the overall average. By applying a logarithmic transformation to smooth out these extremes, our calculations finally revealed the market's true trend (flipping a student $t$-test $p$-value from a deceptive $0.6578$ to a highly significant $3.71 \times 10^{-13}$). Next, our analysis proved that features don’t act alone. For instance, combining high-end exterior finishes with a great overall structure creates a 'premium synergy' that boosts the price ($+0.0906$). Conversely, central air conditioning is a great bonus on a standard house, but becomes a mere baseline expectation in the luxury segment ($-0.1034$). 

Finally, we tested two approaches to predict property values: a classic 9-variable mathematical model that already performed quite well ($R^2 = 0.815$), and an optimized neural network (MLP). By capturing all the subtle market traps and cross-dependencies that standard equations missed, the artificial intelligence easily outperformed our baseline model, delivering an excellent final accuracy (RMSLE of $\sim 0.12689$).

---

## 1. Introduction

The real estate market is a complex subject that requires taking into account a wide range of aspects as well as their interactions. By aspects, we mean geographic, structural, or contextual variables. Consequently, predicting the prices of various properties with surgical precision is not only a fundamental economic challenge but also an essential benchmark for evaluating predictive modeling techniques. Within the framework of the Computational Statistics course taught by Professor Yiyu Lydia Chen, and with the support of teaching assistants Elif Yilmaz and Gert Lek, we will attempt to carry out this project by building, evaluating, and comparing different predictive frameworks based on the famous 'Ames Housing' dataset, compiled by Dean De Cock and hosted on Kaggle.

### 1.1 Project Objectives
Through this project, we aim to achieve several key goals. First, we intend to develop a deep and statistically rigorous understanding of the various factors and variables present in our dataset, primarily by leveraging statistical inference. Subsequently, we will attempt to design and construct advanced machine learning models to develop a predictive engine capable of generalizing to unseen data. This process will include a comparative analysis between results obtained via linear regression and those derived from more flexible neural networks. Our ultimate objective is to optimize our predictive accuracy to achieve the highest possible score in the Kaggle competition, while remaining vigilant against pitfalls such as data leakage, which could lead to evaluation errors or overfitting.

### 1.2 Main Methods and Methodology
To achieve high-precision predictions from raw data, this project follows a comprehensive and rigorous methodology, combining classical statistical methods with advanced non-parametric machine learning approaches:
* **Exploratory Data Analysis & Inference:** Utilization of descriptive statistical tools, such as confidence intervals and parametric hypothesis tests, to validate assumptions regarding population distributions and isolate the significant effects of each variable.
* **Variance and Factorial Analysis:** Through one-way and two-way ANOVA, we seek to measure and quantify the impact of qualitative variables. This is complemented by a $2^k$ factorial design to isolate main effects and understand interactions between features.
* **Parametric Modeling:** We formulated an Ordinary Least Squares (OLS) linear regression model to establish a solid and highly interpretable baseline. This work was accompanied by an in-depth residual diagnostic, allowing us to rigorously verify the absence of heteroscedasticity, multicollinearity, or normality issues within our data.
* **Non-Parametric Modeling:** To transcend the limitations imposed by linearity, we implemented a Multilayer Perceptron (MLP) regressor. This architecture relies on automated preprocessing pipelines via scikit-learn, Z-score standardization, and advanced regularization techniques—specifically L2 regularization and Early Stopping—to ensure the reliability of our learning process.

### 1.3 Analysis Plan
Five is the number of analytical stages designed to represent this methodological progression. We will begin by defining and briefly interpreting our data, in order to start creating automated pipelines aimed at cleaning, normalizing, and refining. Once this is done, we will tackle the portion focused on statistical inference as well as analysis of variance (ANOVA) to identify signals. This set will feed into the $2^k$ factorial design, which will ultimately allow us to precisely identify the interactions between the target variables.
Once this understanding of the context and the structure of our environment is acquired, we will attempt to model all of this by first training the parametric regression, while keeping diagnostic error analysis in mind. Finally, we will focus on training a robust neural network. This stage will involve a 5-fold cross-validation protocol, the goal being that our various validation scores are in alignment with the Kaggle evaluation environment.

---

## 2. Data and Preprocessing

### 2.1 Data Source
Our data originates from the Ames Housing Dataset, compiled by Dean De Cock and made available on Kaggle for comparative regression model analysis. This dataset consists of $N = 1,460$ observations (real estate sales recorded in Ames, Iowa). For each observation, there are 79 explanatory variables, of which 36 are numerical (e.g., GrLivArea, LotArea, TotalBsmtSF) while 43 are classified as qualitative nominal or ordinal variables (e.g., Neighborhood, KitchenQual, MSZoning).
The variable we aim to predict with the highest possible precision is SalePrice, measured in U.S. dollars. A preliminary analysis of the raw target variable reveals a distribution with right-skewness (a long right tail), deviating from a normal distribution ($\mu = 180,921.20$, $\sigma = 79,442.50$, with a range from $\$34,900$ to $\$755,000$). During our initial exploratory analysis, using boxplots and Q-Q plots, it appeared that the dataset did not respect classical parametric assumptions.



### 2.2 Preprocessing Decisions
Exploratory analysis via Boxplots and Quantile-Quantile (Q-Q) plots highlights a major violation of parametric assumptions:
* **Raw Target Space:** First, we observed skewness in sale prices (right-tailed) along with 61 outliers (threshold of $1.5 \times \text{IQR}$). The Q-Q plot supported this, showing a curve deviating and dropping away from the normal line.
* **Logarithmic Transformation:** To rectify this and stabilize variance, we applied a logarithmic transformation: 
  $$y = \log(\text{SalePrice})$$

Following this manipulation, the distribution tightens, allowing us to obtain a more balanced bell curve (a symmetric interval of approximately 10.5 to 13.5). Simultaneously, the number of outliers was drastically reduced from 61 to 28, and naturally, our Q-Q plot now shows much better adherence to the normality line, which is crucial. Thanks to these adjustments, our future neural network will be able to learn from this dataset without being "distracted" or biased by exceptional luxury properties, which would otherwise decrease the overall precision of our predictions.


---

## 3. Classical Statistical Inference

In this section of the project, we transition from simple data manipulation to formal statistical inference. Based on our sample of 1,460 properties, we calculate point estimators and establish robust confidence intervals to gain a clearer overview of the true population parameters. Furthermore, we test our working hypotheses using parametric hypothesis tests to evaluate specific claims about the market.

### 3.1 Descriptive Statistics
To establish an operational foundation, we examine the mean ($\bar{X}$), variance ($S^2$), and standard deviation ($S$) of our target variable, SalePrice. This analysis also includes five other significant structural variables: ground living area (GrLivArea), overall material and finish quality (OverallQual), lot size (LotArea), construction year (YearBuilt), and total basement area (TotalBsmtSF). 

**Table 1: Descriptive Statistical Summary of Key Housing Features ($n = 1460$)**

| Feature | Sample Mean ($\bar{X}$) | Sample Variance ($S^2$) | Std Deviation ($S$) |
| :--- | :--- | :--- | :--- |
| `SalePrice` ($) | $180,921.20$ | $6,311,111,264.30$ | $79,442.50$ |
| `GrLivArea` (sqft) | $1,515.46$ | $276,129.63$ | $525.48$ |
| `OverallQual` (1-10) | $6.10$ | $1.91$ | $1.38$ |
| `LotArea` (sqft) | $10,516.83$ | $99,625,649.65$ | $9,981.26$ |
| `YearBuilt` (year) | $1,971.27$ | $912.22$ | $30.20$ |
| `TotalBsmtSF` (sqft) | $1,057.43$ | $192,462.36$ | $438.71$ |

The empirical findings point to structural patterns:
* **High Dispersion:** A significant dispersion is observed in our target variable, SalePrice ($S = \$79,442.50$), indicating a market characterized by substantial economic disparity.
* **Physical Volatility:** Physical features such as LotArea reveal significant volatility ($S \approx 9,981 \text{ square feet}$), driven by properties often located on the periphery, which considerably stretch the raw distribution to the right.
* **Ordinal Consistency:** Qualitative indicators, such as OverallQual, appear to be symmetrically concentrated around a mean of $6.10 / 10$. This can be considered a reliable ordinal benchmark for our future modeling.

### 3.2 Confidence Interval
Since the true population variance ($\sigma^2$) is unknown, we must construct a confidence interval for the population mean ($\mu$) using the Student's t-distribution, in accordance with the Central Limit Theorem:

$$CI_{1-\alpha} = \left[ \bar{X} - t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}, \bar{X} + t_{\alpha/2, n-1} \frac{S}{\sqrt{n}} \right]$$

By leveraging our sample statistics, we calculated 95% and 99% confidence intervals for the true mean house price ($\mu_{\text{price}}$), as well as for our most representative continuous variables. This comparison allows us to visualize the impact of the confidence level on the precision of our estimates:

**Table 2: Population Mean Confidence Interval Analysis**

| Feature | 95% Confidence Interval | 99% Confidence Interval |
| :--- | :--- | :--- |
| `SalePrice` | $[\$176,842.84, \,\, \$184,999.55]$ | $[\$175,558.76, \,\, \$186,283.63]$ |
| `GrLivArea` | $[1,488.49, \,\, 1,542.44]$ | $[1,479.99, \,\, 1,550.93]$ |
| `OverallQual` | $[6.03, \,\, 6.17]$ | $[6.01, \,\, 6.19]$ |
| `LotArea` | $[10,004.42, \,\, 11,029.24]$ | $[9,843.08, \,\, 11,190.57]$ |
| `TotalBsmtSF` | $[1,034.91, \,\, 1,079.95]$ | $[1,027.82, \,\, 1,087.04]$ |

This analysis highlights the fundamental statistical trade-off between interval precision and the desired degree of certainty. Indeed, as we increase our confidence coefficient from 95% to 99%, the critical $t$-value rises ($t_{0.025} \approx 1.962$ increases to $t_{0.005} \approx 2.579$). The direct consequence is that the interval widens, increasing from an amplitude of $\$8,156.71$ to $\$10,724.87$ for *SalePrice*.

### 3.3 Hypothesis Test
Regarding our hypothesis testing, we will perform a one-sample two-tailed t-test to evaluate a significant hypothesis: is the true mean sale price of residential properties in Ames ($\mu$) significantly different from $\mu_0 = \$180,000$?

#### 3.3.1 Hypothesis Formulation
$$H_0: \mu = 180,000, \quad H_1: \mu \neq 180,000$$
The null hypothesis ($H_0$) posits that the true mean transaction price is $\$180,000$, whereas the alternative hypothesis ($H_1$) asserts that there is a statistically significant difference.

#### 3.3.2 Test Statistic and Decision Rule
Given the size of our dataset ($n = 1,460$), the test statistic follows a Student's t-distribution:
$$t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}}$$
Substituting our empirical values ($\bar{X} = 180,921.20$, $S = 79,442.50$, $\sqrt{1460} \approx 38.21$):
$$t = \frac{180,921.20 - 180,000}{79,442.50 / 38.21} = \frac{921.20}{2,079.10} \approx 0.443$$
The two-tailed p-value associated with $t = 0.443$ and $df = 1,459$ is estimated at: $p\text{-value} = 0.6578$

#### 3.3.3 Statistical Interpretation
Compared to standard significance levels ($\alpha = 0.05$ and $\alpha = 0.01$), our p-value ($0.6578$) is significantly higher than $\alpha$; therefore, we fail to reject the null hypothesis ($H_0$).
The difference of $\$921.20$ between our sample mean and the theoretical reference is statistically negligible and lies within the limits of random sampling variation. We conclude that there is insufficient statistical evidence to state that the true mean residential sale price in Ames differs from $\$180,000$.

#### 3.3.4 Log-Transformed Hypothesis Testing
To evaluate how the underlying data distribution impacts our statistical inferences, we repeat the same one-sample t-test framework on the log-transformed target variable. The theoretical target baseline must be shifted to the logarithmic scale: $\mu_{0\,\text{log}} = \log(180,000) \approx 12.1007$.

We formalize our hypotheses for this transformed space as follows:
$$H_0: \mu_{\text{log}} = 12.1007, \quad H_1: \mu_{\text{log}} \neq 12.1007$$

By substituting the logged empirical values generated from our sample ($\bar{X}_{\text{log}} = 12.0241$ and $S_{\text{log}} = 0.3995$), the test yields a highly significant t-statistic:
$$t = \frac{12.0241 - 12.1007}{0.3995 / \sqrt{1460}} \approx -7.3331$$

The corresponding two-tailed $p$-value is estimated at $3.7148 \times 10^{-13}$. Given that this $p$-value is drastically lower than any standard significance threshold ($\alpha = 0.05$ or $\alpha = 0.01$), we **strongly reject the null hypothesis ($H_0$)**.

This complete reversal of the statistical decision highlights the profound impact of right-skewness and extreme outliers on classical inference. On the raw scale, a handful of exceptional luxury properties artificially pulled the arithmetic mean upward, making it appear statistically close to the $\$180,000$ baseline. Once the variance is stabilized through the logarithmic transformation, the true central tendency of the market is revealed, placing the theoretical value of $12.1007$ entirely outside our empirical $95\%$ confidence interval ($[12.0036, 12.0446]$).

---

## 4. ANOVA for Ordinal Features

To identify significant variables that concretely influence a property's price and to reduce the dimensionality of our data, we employ Analysis of Variance (ANOVA). This analytical approach indicates whether a variable (categorical, ordinal, or nominal) has a significant impact on our (transformed) target variable, $y = \log(\text{SalePrice})$.

### 4.1 Features Analyzed
As specified, we applied this method across 10 key variables covering structural quality, qualitative material ratings, geographic/topographic indicators, and temporal markers. For certain variables, missing data is synonymous with the absence of the feature (notably for BsmtQual and FireplaceQu); we therefore encoded these cases as a specific category, "None," before proceeding with hypothesis testing.

**Table 3: Categorical and Ordinal Features Selected for ANOVA Screening**

| # | Feature Label | Observed Levels | Qualitative Description |
| :--- | :--- | :--- | :--- |
| 1 | `OverallQual` | 1 to 10 | Overall material and finish quality rating |
| 2 | `ExterQual` | Po, Fa, TA, Gd, Ex | Quality rating of exterior materials |
| 3 | `BsmtQual` | None, Po, Fa, TA, Gd, Ex | Evaluates the height/volume of the basement |
| 4 | `KitchenQual` | Po, Fa, TA, Gd, Ex | Interior kitchen finish quality rating |
| 5 | `FireplaceQu` | None, Po, Fa, TA, Gd, Ex | Fireplace structural quality rating |
| 6 | `CentralAir` | N, Y | Presence of a central air conditioning system |
| 7 | `LotShape` | IR3, IR2, IR1, Reg | General geometric shape of the property parcel |
| 8 | `LandSlope` | Sev, Mod, Gtl | Slope gradient of the land terrain |
| 9 | `MoSold` | 1 to 12 | Calendar month of transaction entry |
| 10 | `YrSold` | 2006 to 2010 | Calendar year of transaction execution |

### 4.2 One-Way ANOVA
For each identified characteristic $j$, we apply a one-way ANOVA. This method allows us to determine if the variations in our target variable, $\log(\text{SalePrice})$, are statistically attributable to the different levels of these characteristics. We formalize this approach with the following hypotheses:

$$H_0 : \mu_1 = \mu_2 = \dots = \mu_k \quad \text{vs.} \quad H_1 : \exists (i, m) \text{ s.t. } \mu_i \neq \mu_m$$

The test statistic evaluates the ratio between the variance explained between groups and the variance recorded within groups:

$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}} = \frac{SS_{\text{between}} / (k - 1)}{SS_{\text{within}} / (n - k)}$$

Using a critical alpha threshold of $\alpha = 0.05$, the empirical computations generated from our sample yield the following hierarchy of feature significance:

**Table 4: Empirical One-Way ANOVA Screening Results ($\alpha = 0.05$)**

| Tested Feature | Degrees of Freedom ($df$) | $F$-Statistic | $p$-Value | Statistical Status |
| :--- | :--- | :--- | :--- | :--- |
| `ExterQual` | 3 | $415.30$ | $6.93 \times 10^{-195}$ | Highly Significant |
| `KitchenQual` | 3 | $393.32$ | $4.44 \times 10^{-187}$ | Highly Significant |
| `OverallQual` | 9 | $332.17$ | $0.00 \times 10^{0}$ | Highly Significant |
| `BsmtQual` | 4 | $300.39$ | $2.03 \times 10^{-188}$ | Highly Significant |
| `CentralAir` | 1 | $205.67$ | $9.86 \times 10^{-44}$ | Highly Significant |
| `FireplaceQu` | 5 | $131.20$ | $6.96 \times 10^{-115}$ | Highly Significant |
| `LotShape` | 3 | $46.73$ | $7.86 \times 10^{-29}$ | Highly Significant |
| `LandSlope` | 2 | $1.08$ | $3.39 \times 10^{-1}$ | Not Significant |
| `MoSold` | 11 | $0.99$ | $4.50 \times 10^{-1}$ | Not Significant |
| `YrSold` | 4 | $0.74$ | $5.66 \times 10^{-1}$ | Not Significant |



### 4.3 Two-Way ANOVA and Interactions
To determine and verify whether the impact of our significant predictors is merely additive or masks more complex dependencies, we also employed a two-way ANOVA by integrating explicit interaction terms:
$$y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$$
Out of our 21 combinations, our analyses reveal 18 with highly significant interactions. Only three exceptions show no notable synergy:
1. **`KitchenQual` $\times$ `CentralAir`:** displays an unadjusted joint p-value of $0.7787$, revealing no structural interaction.
2. **`FireplaceQu` $\times$ `LotShape`:** Yields a $p$-value of $0.6297$.
3. **`CentralAir` $\times$ `LotShape`:** Yields a $p$-value of $0.4169$.

As for the other 18 pairs, their p-values are close to 0, confirming a strong "synergy." For instance, the interaction between ExterQual and OverallQual ($p \approx 0.0000$) reveals a significant multiplier effect. It reveals, in particular, that the Ames dataset is saturated with cross-dependencies, which justifies the use of non-linear layers in our future neural network.

---

## 5. $2^k$ Factorial Design

To go further and understand in detail the structural mechanisms that influence prices (beyond simple additive correlations), we implemented a $2^3$ factorial design. 

### 5.1 Factor Selection and Binarization
Based on our preliminary analysis (ANOVA), we isolate three key orthogonal dimensions:
* **Factor A — Central Air Conditioning (`CentralAir`):** Binary state (absence $-1$ / presence $+1$).
* **Factor B — Overall Structural Quality (`OverallQual`):** Binarized using an objective threshold of 7. Properties with a rating $< 7$ ($-1$) form our "Standard" segment, while those $\ge 7$ ($+1$) represent the "Premium/Luxury" segment.
* **Factor C — Exterior Quality (`ExterQual`):** Binarized into two strategic segments: Budget ($-1$) and Premium ($+1$).

**Table 5: Design Matrix and Cell Group Means for $\log(\text{SalePrice})$**

| Comb. | Factor A (`CentralAir`) | Factor B (`OverallQual`) | Factor C (`ExterQual`) | Mean Log-Price ($\bar{y}_{ijk}$) |
| :---: | :---: | :---: | :---: | :---: |
| $(1)$ | $-1$ | $-1$ | $-1$ | $11.4515$ |
| $c$ | $-1$ | $-1$ | $+1$ | $11.6335$ |
| $b$ | $-1$ | $+1$ | $-1$ | $11.8635$ |
| $bc$ | $-1$ | $+1$ | $+1$ | $12.3610$ |
| $a$ | $+1$ | $-1$ | $-1$ | $11.8284$ |
| $ac$ | $+1$ | $-1$ | $+1$ | $12.0390$ |
| $ab$ | $+1$ | $+1$ | $-1$ | $12.1681$ |
| $abc$ | $+1$ | $+1$ | $+1$ | $12.4253$ |

### 5.2 Main Effects and Interactions
We decomposed the variance using Yates' algebraic contrasts.

#### 5.2.1 Quantification of Structural Contrasts
The main effects and interaction contrasts extracted from our design matrix are calculated as follows:
* **Main Effect of Factor A (Central Air):** $\text{Effect}_A = +0.2878$ log-points.
* **Main Effect of Factor B (Overall Quality):** $\text{Effect}_B = +0.4663$ log-points.
* **Main Effect of Factor C (Exterior Quality):** $\text{Effect}_C = +0.2868$ log-points.
* **Interaction Effect (AB):** $\text{Effect}_{AB} = -0.1034$ log-points.
* **Interaction Effect (AC):** $\text{Effect}_{AC} = -0.0529$ log-points.
* **Interaction Effect (BC):** $\text{Effect}_{BC} = +0.0906$ log-points.
* **Interaction Effect (ABC):** $\text{Effect}_{ABC} = -0.0672$ log-points.

#### 5.2.2 Economic Interpretation
Analyzing the main effects, `OverallQual` (Factor B) emerges as the undisputed primary driver of property valuation ($+0.4663$), doing the bulk of the heavy lifting in our model's predictions.

The interaction effects, however, present a more nuanced market dynamic:
* **Threshold of Expectations (AB = $-0.1034$):** In premium homes, climate control is viewed as a baseline necessity rather than a luxury value-add, diminishing its relative premium. Conversely, introducing central air conditioning to a lower-tier property significantly spikes
