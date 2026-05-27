# Predicting House Sale Prices with Statistical Modeling and Neural Networks
**Project Report**

Student 1: Ali Mohamed Saleh [24502783]  
Student 2: Haddad Youssef Bilal [2383725]  

Computational Statistics, Spring 2026, submitted on May 27, 2026

---

## Abstract

Predicting house prices often feels like hitting a moving target: a home isn't just a simple sum of rooms, but a complex mix of features and market psychology. This project dissects the Ames housing dataset by combining classical statistics with artificial intelligence. Right from the start, we faced a major hurdle: a handful of overpriced luxury homes were skewing the overall average. By applying a logarithmic transformation to smooth out these extremes, our calculations finally revealed the market's true trend (flipping a student t-test p-value from a deceptive 0.6578 to a highly significant $3.71\times10^{-13}$). Next, our analysis proved that features don't act alone. For instance, combining high-end exterior finishes with a great overall structure creates a 'premium synergy' that boosts the price (+0.0906). Conversely, central air conditioning is a great bonus on a standard house, but becomes a mere baseline expectation in the luxury segment (-0.1034). Finally, we tested two approaches to predict property values: a classic 9-variable mathematical model that already performed quite well ($R^{2}=0.815$), and an optimized neural network (MLP). By capturing all the subtle market traps and cross-dependencies that standard equations missed, the artificial intelligence easily outperformed our baseline model, delivering an excellent final accuracy (RMSLE of ~ 0.12689).

---

## Contents
1. Introduction
   1.1 Project Objectives
   1.2 Main Methods and Methodology
   1.3 Analysis Plan
2. Data and Preprocessing
   2.1 Data Source
   2.2 Preprocessing Decisions
3. Classical Statistical Inference
   3.1 Descriptive Statistics
   3.2 Confidence Interval
   3.3 Hypothesis Test
      3.3.1 Hypothesis Formulation
      3.3.2 Test Statistic and Decision Rule
      3.3.3 Statistical Interpretation
      3.3.4 Log-Transformed Hypothesis Testing
4. ANOVA for Ordinal Features
   4.1 Features Analyzed
   4.2 One-Way ANOVA
   4.3 Two-Way ANOVA and Interactions
5. 2k Factorial Design
   5.1 Factor Selection and Binarization
   5.2 Main Effects and Interactions
      5.2.1 Quantification of Structural Contrasts
      5.2.2 Economic Interpretation
6. Parametric Regression
   6.1 Model Specification
   6.2 Regression Results
   6.3 Model Diagnostics
7. Neural-Network Regression
   7.1 Model Inputs and Preprocessing
      7.1.1 Outlier Removal and Feature Engineering
      7.1.2 Addressing Skewness and Target Transformation
      7.1.3 Imputation and Final Matrix Assembly
   7.2 Training and Hyperparameter Tuning
      7.2.1 The Hyperparameter Grid
      7.2.2 Cross-Validation and Final MLP
   7.3 Final Model Evaluation
   7.4 Kaggle Submission File

---

## 1 Introduction

The real estate market is a complex subject that requires taking into account a wide range of aspects as well as their interactions. By aspects, we mean geographic, structural, or contextual variables. Consequently, predicting the prices of various properties with surgical precision is not only a fundamental economic challenge but also an essential benchmark for evaluating predictive modeling techniques. Within the framework of the Computational Statistics course taught by Professor Yiyu Lydia Chen, and with the support of teaching assistants Elif Yilmaz and Gert Lek, we will attempt to carry out this project by building, evaluating, and comparing different predictive frameworks based on the famous 'Ames Housing' dataset, compiled by Dean De Cock and hosted on Kaggle.

### 1.1 Project Objectives

Through this project, we aim to achieve several key goals. First, we intend to develop a deep and statistically rigorous understanding of the various factors and variables present in our dataset, primarily by leveraging statistical inference. Subsequently, we will attempt to design and construct advanced machine learning models to develop a predictive engine capable of generalizing to unseen data. This process will include a comparative analysis between results obtained via linear regression and those derived from more flexible neural networks. Our ultimate objective is to optimize our predictive accuracy to achieve the highest possible score in the Kaggle competition, while remaining vigilant against pitfalls such as data leakage, which could lead to evaluation errors or overfitting.

### 1.2 Main Methods and Methodology

To achieve high-precision predictions from raw data, this project follows a comprehensive and rigorous methodology, combining classical statistical methods with advanced non-parametric machine learning approaches:

* Exploratory Data Analysis & Inference: Utilization of descriptive statistical tools, such as confidence intervals and parametric hypothesis tests, to validate assumptions regarding population distributions and isolate the significant effects of each variable.
* Variance and Factorial Analysis: Through one-way and two-way ANOVA, we seek to measure and quantify the impact of qualitative variables. This is complemented by a $2^{k}$ factorial design to isolate main effects and understand interactions between features.
* Parametric Modeling: We formulated an Ordinary Least Squares (OLS) linear regression model to establish a solid and highly interpretable baseline. This work was accompanied by an in-depth residual diagnostic, allowing us to rigorously verify the absence of heteroscedasticity, multicollinearity, or normality issues within our data.
* Non-Parametric Modeling: To transcend the limitations imposed by linearity, we implemented a Multilayer Perceptron (MLP) regressor. This architecture relies on automated preprocessing pipelines via scikit-learn, Z-score standardization, and advanced regularization techniques specifically L2 regularization and Early Stopping to ensure the reliability of our learning process.

### 1.3 Analysis Plan

Five is the number of analytical stages designed to represent this methodological progression. We will begin by defining and briefly interpreting our data, in order to start creating automated pipelines aimed at cleaning, normalizing, and refining. Once this is done, we will tackle the portion focused on statistical inference as well as analysis of variance (ANOVA) to identify signals. This set will feed into the $2^{k}$ factorial design, which will ultimately allow us to precisely identify the interactions between the target variables. Once this understanding of the context and the structure of our environment is acquired, we will attempt to model all of this by first training the parametric regression, while keeping diagnostic error analysis in mind. Finally, we will focus on training a robust neural network. This stage will involve a 5-fold cross-validation protocol, the goal being that our various validation scores are in alignment with the Kaggle evaluation environment.

---

## 2 Data and Preprocessing

### 2.1 Data Source

Our data originates from the Ames Housing Dataset, compiled by Dean De Cock and made available on Kaggle for comparative regression model analysis. This dataset consists of $N=$ 1, 460 observations (real estate sales recorded in Ames, Iowa). For each observation, there are 79 explanatory variables, of which 36 are numerical (e.g., GrLivArea, LotArea, TotalBsmtSF) while 43 are classified as qualitative nominal or ordinal variables (e.g., Neighborhood, KitchenQual, MSZoning). The variable we aim to predict with the highest possible precision is SalePrice, measured in U.S. dollars. A preliminary analysis of the raw target variable reveals a distribution with right-skewness (a long right tail), deviating from a normal distribution ($\mu=180,921.20$, $\sigma=79,442.50$, with a range from 34,900 to 755,000). During our initial exploratory analysis, using boxplots and Q-Q plots, it appeared that the dataset did not respect classical parametric assumptions:

![Figure 1: Distribution of the variable "Sale Price" with skewness]()

### 2.2 Preprocessing Decisions

Exploratory analysis via Boxplots and Quantile-Quantile (Q-Q) plots highlights a major violation of parametric assumptions:

* Raw Target Space: First, we observed skewness in sale prices (right-tailed) along with 61 outliers (threshold of $1.5\times10R$. The Q-Q plot supported this, showing a curve deviating and dropping away from the normal line.
* Logarithmic Transformation: To rectify this and stabilize variance, we applied a logarithmic transformation:
  $y = \log(SalePrice)$

Following this manipulation, the distribution tightens, allowing us to obtain a more balanced bell curve (a symmetric interval of approximately 10.5 to 13.5). Simultaneously, the number of outliers was drastically reduced from 61 to 28, and naturally, our Q-Q plot now shows much better adherence to the normality line, which is crucial. Thanks to these adjustments, our future neural network will be able to learn from this dataset without being "distracted" or biased by exceptional luxury properties, which would otherwise decrease the overall precision of our predictions.

![Figure 2: Distribution of the variable log("SalePrice") with less skewness]()

---

## 3 Classical Statistical Inference

In this section of the project, we transition from simple data manipulation to formal statistical inference. Based on our sample of 1,460 properties, we calculate point estimators and establish robust confidence intervals to gain a clearer overview of the true population parameters. Furthermore, we test our working hypotheses using parametric hypothesis tests to evaluate specific claims about the market.

### 3.1 Descriptive Statistics

To establish an operational foundation, we examine the mean (X), variance $(S^{2})$ and standard deviation (S) of our target variable, SalePrice. This analysis also includes five other significant structural variables: ground living area (GrLivArea), overall material and finish quality (OverallQual), lot size (LotArea), construction year (YearBuilt), and total basement area (TotalBsmtSF). The empirical results in Table 1 highlight structural trends:

**Table 1: Descriptive Statistical Summary of Key Housing Features (n=1460)**

| Feature | Sample Mean ( ) | Sample Variance $(S^{2})$ | Std Deviation (S) |
|---|---|---|---|
| SalePrice (\$) | 180,921.20 | 6,311, 111, 264.30 | 79,442.50 |
| GrLivArea (sqft) | 1,515.46 | 276, 129.63 | 525.48 |
| OverallQual (1-10) | 6.10 | 1.91 | 1.38 |
| LotArea (sqft) | 10,516.83 | 99, 625, 649.65 | 9,981.26 |
| YearBuilt (year) | 1,971.27 | 912.22 | 30.20 |
| TotalBsmtSF (sqft) | 1,057.43 | 192, 462.36 | 438.71 |

The empirical findings in Table 1 point to structural patterns:
* High Dispersion: A significant dispersion is observed in our target variable, SalePrice $(S=$ \$79, 442.50), indicating a market characterized by substantial economic disparity.
* Physical Volatility: Physical features such as LotArea reveal significant volatility ($S\approx$ 9,981 square feet), driven by properties often located on the periphery, which considerably stretch the raw distribution to the right.
* Ordinal Consistency: Qualitative indicators, such as OverallQual, appear to be symmetrically concentrated around a mean of $6.10/10$. This can be considered a reliable ordinal benchmark for our future modeling.

### 3.2 Confidence Interval

Since the true population variance $(\sigma^{2})$ is unknown, we must construct a confidence interval for the population mean ( ) using the Student's t-distribution, in accordance with the Central Limit Theorem:

$$CI_{1-\alpha}=[\overline{X}-t_{\alpha/2,n-1}\frac{S}{\sqrt{n}},\overline{X}+t_{\alpha/2,n-1}\frac{S}{\sqrt{n}}]$$

where $t_{\alpha/2,n-1}$ represents the critical value with $n-1=1,459$ degrees of freedom at a significance level a. By leveraging our sample statistics, we calculated 95% and 99% confidence intervals for the true mean house price $(\mu_{price})$, as well as for our most representative continuous variables. This comparison allows us to visualize the impact of the confidence level on the precision of our estimates:

**Table 2: Population Mean Confidence Interval Analysis**

| Feature | 95% Confidence Interval | 99% Confidence Interval |
|---|---|---|
| SalePrice | [\$176,842.84, \$184,999.55] | [\$175,558.76, \$186, 283.63] |
| GrLivArea | [1,488.49, 1,542.44] | [1,479.99, 1,550.93] |
| OverallQual | [6.03, 6.17] | [6.01, 6.19] |
| LotArea | [10,004.42, 11,029.24] | [9,843.08, 11, 190.57] |
| TotalBsmtSF | [1,034.91, 1,079.95] | [1,027.82, 1,087.04] |

This analysis highlights the fundamental statistical trade-off between interval precision and the desired degree of certainty. Indeed, as we increase our confidence coefficient from 95% to 99%, the critical t-value rises $(t_{0.025}\approx1.962$ increases to $t_{0.005}\approx2.579)$ The direct consequence is that the interval widens, increasing from an amplitude of \$8, 156.71 to \$10, 724.87 for SalePrice. This mechanism perfectly illustrates the balance that every analyst must manage: we obtain a stronger probabilistic assurance of capturing the true population value, but at the cost of a wider estimation, which is therefore mathematically less precise.

### 3.3 Hypothesis Test

Regarding our hypothesis testing, we will perform a one-sample two-tailed t-test to evaluate a significant hypothesis: is the true mean sale price of residential properties in Ames ( ) significantly different from $\mu_{0}=\$180,000?$

#### 3.3.1 Hypothesis Formulation
$H_{0}:\mu=180,000$, $H_{1}:\mu\ne180,000$

The null hypothesis $(H_{0})$ posits that the true mean transaction price is \$180,000, whereas the alternative hypothesis $(H_{1})$ asserts that there is a statistically significant difference.

#### 3.3.2 Test Statistic and Decision Rule
Given the size of our dataset $(n=1,460)$, the test statistic follows a Student's t-distribution:
$$t=\frac{\overline{X}-\mu_{0}}{S/\sqrt{n}}$$
Substituting our empirical values $(\overline{X}=180,921.20$, $S=79,442.50$, $\sqrt{1460}\approx38.21)$:
$$t=\frac{180,921.20-180,000}{79,442.50/38.21}=\frac{921.20}{2,079.10}\approx0.443$$
The two-tailed p-value associated with $t=0.443$ and $df=1,459$ is estimated at: p-value = 0.6578

#### 3.3.3 Statistical Interpretation
Compared to standard significance levels $(\alpha=0.05$ and $\alpha=0.01)$, our p-value (0.6578) is significantly higher than ; therefore, we fail to reject the null hypothesis $(H_{0})$. The difference of \$921.20 between our sample mean and the theoretical reference is statistically negligible and lies within the limits of random sampling variation. We conclude that there is insufficient statistical evidence to state that the true mean residential sale price in Ames differs from \$180,000.

#### 3.3.4 Log-Transformed Hypothesis Testing
To evaluate how the underlying data distribution impacts our statistical inferences, we repeat the same one-sample t-test framework on the log-transformed target variable. The theoretical target baseline must be shifted to the logarithmic scale: $\mu_{0~log}=log(180,000)\approx12.1007$
We formalize our hypotheses for this transformed space as follows:
$H_{0}:\mu_{log}=12.1007$, $H_{1}:\mu_{log}\ne12.1007$

By substituting the logged empirical values generated from our sample $(X_{log}=12.0241$ and $S_{log}=0.3995)$, the test yields a highly significant t-statistic:
$$t=\frac{12.0241-12.1007}{0.3995/\sqrt{1460}}\approx-7.3331$$
The corresponding two-tailed p-value is estimated at $3.7148\times10^{-13}$. Given that this p-value is drastically lower than any standard significance threshold $(\alpha=0.05$ or $\alpha=0.01$, we strongly reject the null hypothesis $(H_{0})$.

This complete reversal of the statistical decision highlights the profound impact of right-skewness and extreme outliers on classical inference. On the raw scale, a handful of exceptional luxury properties artificially pulled the arithmetic mean upward, making it appear statistically close to the \$180,000 baseline. Once the variance is stabilized through the logarithmic transformation, the true central tendency of the market is revealed, placing the theoretical value of 12.1007 entirely outside our empirical 95% confidence interval ([12.0036, 12.0446]).

---

## 4 ANOVA for Ordinal Features

To identify significant variables that concretely influence a property's price and to reduce the dimensionality of our data, we employ Analysis of Variance (ANOVA). This analytical approach indicates whether a variable (categorical, ordinal, or nominal) has a significant impact on our (transformed) target variable, y  log(SalePrice).

### 4.1 Features Analyzed
As specified, we applied this method across 10 key variables covering structural quality, qualitative material ratings, geographic/topographic indicators, and temporal markers. For certain variables, missing data is synonymous with the absence of the feature (notably for BsmtQual and FireplaceQu); we therefore encoded these cases as a specific category, "None," before proceeding with hypothesis testing.

**Table 3: Categorical and Ordinal Features Selected for ANOVA Screening**

| # | Feature Label | Observed Levels | Qualitative Description |
|---|---|---|---|
| 1 | OverallQual | 1 to 10 | Overall material and finish quality rating |
| 2 | ExterQual | Po, Fa, TA, Gd, Ex | Quality rating of exterior materials |
| 3 | BsmtQual | None, Po, Fa, TA, Gd, Ex | Evaluates the height/volume of the basement |
| 4 | KitchenQual | Po, Fa, TA, Gd, Ex | Interior kitchen finish quality rating |
| 5 | FireplaceQu | None, Po, Fa, TA, Gd, Ex | Fireplace structural quality rating |
| 6 | CentralAir | N, Y | Presence of a central air conditioning system |
| 7 | LotShape | IR3, IR2, IR1, Reg | General geometric shape of the property parcel |
| 8 | LandSlope | Sev, Mod, Gtl | Slope gradient of the land terrain |
| 9 | MoSold | 1 to 12 | Calendar month of transaction entry |
| 10 | YrSold | 2006 to 2010 | Calendar year of transaction execution |

### 4.2 One-Way ANOVA

For each identified characteristic j, we apply a one-way ANOVA. This method allows us to determine if the variations in our target variable, log(SalePrice), are statistically attributable to the different levels of these characteristics. Concretely, we verify if splitting the sample into sub-groups produces significantly distinct means. We formalize this approach with the following hypotheses:

$$H_{0}:\mu_{1}=\mu_{2}=\cdot\cdot\cdot=\mu_{k} \text{ VS. } H_{1}:\exists(i,m) \text{ s.t. } \mu_{i}\ne\mu_{m}$$

where $\mu_{k}$ represents the true population mean of log(SalePrice) within level k of a given characteristic. The test statistic evaluates the ratio between the variance explained between groups and the variance recorded within groups:

$$F=\frac{MS_{between}}{MS_{within}}=\frac{SS_{between}/(k-1)}{SS_{within}/(n-k)}$$

Using a critical alpha threshold of $\alpha=0.05$, the empirical computations generated from our sample yield the following hierarchy of feature significance:

**Table 4: Empirical One-Way ANOVA Screening Results ($\alpha=0.05$)**

| Tested Feature | Degrees of Freedom $(df)$ | F-Statistic | p-Value | Statistical Status |
|---|---|---|---|---|
| ExterQual | 3 | 415.30 | $6.93\times10^{-195}$ | Highly Significant |
| KitchenQual | 3 | 393.32 | $4.44\times10^{-187}$ | Highly Significant |
| OverallQual | 9 | 332.17 | $0.00\times10^{0}$ | Highly Significant |
| BsmtQual | 4 | 300.39 | $2.03\times10^{-188}$ | Highly Significant |
| CentralAir | 1 | 205.67 | $9.86\times10^{-44}$ | Highly Significant |
| FireplaceQu | 5 | 131.20 | $6.96\times10^{-115}$ | Highly Significant |
| LotShape | 3 | 46.73 | $7.86\times10^{-29}$ | Highly Significant |
| LandSlope | 2 | 1.08 | $3.39\times10^{-1}$ | Not Significant |
| MoSold | 11 | 0.99 | $4.50\times10^{-1}$ | Not Significant |
| YrSold | 4 | 0.74 | $5.66\times10^{-1}$ | Not Significant |

![Figure 3: Feature Impact on Sale Price via One-Way ANOVA F-Statistics]()

### 4.3 Two-Way ANOVA and Interactions

To determine and verify whether the impact of our significant predictors is merely additive or masks more complex dependencies, we also employed a two-way ANOVA by integrating explicit interaction terms. For any pair of selected significant characteristics, the dependent variable is modeled as follows:
$$Y_{ijk} = \mu + \alpha_{i} + \beta_{j} + (\alpha\beta)_{ij} + \epsilon_{ijk}$$
where $\alpha_{i}$ represents the main effect of factor 1, $\beta_{j}$ the main effect of factor 2, and $(\alpha\beta)_{ij}$ the interaction component. Out of our 21 combinations, our analyses reveal 18 with highly significant interactions. Only three exceptions show no notable synergy:

1. KitchenQual CentralAir: displays an unadjusted joint p-value of 0.7787, revealing no structural interaction.
2. FireplaceQux LotShape: Yields a p-value of 0.6297.
3. CentralAir LotShape: Yields a p-value of 0.4169.

As for the other 18 pairs, their p-values are close to 0, confirming a strong "synergy." For instance, the interaction between ExterQual and OverallQual $(p\approx0.0000)$ reveals a significant multiplier effect: the real estate value premium linked to superior quality exterior materials accelerates non-linearly when applied to a structure already highly rated for its overall construction quality. This way of "mapping" our pairs is important for the remainder of the project. It reveals, in particular, that the Ames dataset is saturated with cross-dependencies. It is precisely this observation that justifies, in our view, the use of non-linear layers in our future neural network, where a classical linear approach would fail to capture the full subtlety of the market.

---

## 5 $2^{k}$ Factorial Design

To go further and understand in detail the structural mechanisms that influence prices (beyond simple additive correlations), we implemented a $2^{3}$ factorial design. This approach allows us to decompose the total variance by isolating the main effects of each factor, thereby highlighting non-linear interactions between structural elements and quality on our transformed target variable, y log(SalePrice).

### 5.1 Factor Selection and Binarization

Based on our preliminary analysis (ANOVA), we isolate three key orthogonal dimensions:
* Factor A Central Air Conditioning (CentralAir): Binary state (absence -1 [No AC]/ presence +1 [With AC]).
* Factor B Overall Structural Quality (OverallQual): Binarized using an objective threshold of 7. Properties with a rating <7 (-1) form our "Standard" segment, while those $\ge7(+1)$ represent the "Premium/Luxury" segment.
* Factor C Exterior Quality (ExterQual): Binarized into two strategic segments: Budget (-1) and Premium (+1).

This design ensures an orthogonal matrix, allowing us to estimate each effect without interference from multicollinearity. The empirical group means $(\overline{y}_{ijk})$ are summarized in Table 5.

**Table 5: Design Matrix and Cell Group Means for log(SalePrice)**

| Comb. | Factor A (CentralAir) | Factor B (OverallQual) | Factor C (ExterQual) | Mean Log-Price $(\overline{y}_{ijk})$ |
|---|---|---|---|---|
| (1) | -1 | -1 | -1 | 11.4515 |
| c | -1 | -1 | +1 | 11.6335 |
| b | -1 | +1 | -1 | 11.8635 |
| bc | -1 | +1 | +1 | 12.3610 |
| a | +1 | -1 | -1 | 11.8284 |
| ac | +1 | -1 | +1 | 12.0390 |
| ab | +1 | +1 | -1 | 12.1681 |
| abc | +1 | +1 | +1 | 12.4253 |

### 5.2 Main Effects and Interactions

We decomposed the variance using Yates' algebraic contrasts. This allows us to quantify the marginal premium of each variable while isolating the interaction effects to evaluate whether the features act synergistically or tend to neutralize each other.

#### 5.2.1 Quantification of Structural Contrasts

An examination of the summary table reveals a distinct valuation trajectory from the baseline to the premium tier. Properties at the lowest factor levels establish a baseline log-price of 11.4515, whereas homes maximized across all three dimensions reach an average of 12.4253. Given the logarithmic scale of our dependent variable, this near 1.0 unit differential translates to a massive exponential increase in actual market value. The main effects and interaction contrasts extracted from our design matrix are calculated as follows:
* Main Effect of Factor A (Central Air): $Effect_{A}=+0.2878~log-points.$
* Main Effect of Factor B (Overall Quality): Effect $B=+0.4663$ log-points.
* Main Effect of Factor C (Exterior Quality): Effectc = +0.2868 log-points.
* Interaction Effect (AB): Effect $k_{AB}=-0.1034~log-points$.
* Interaction Effect (AC): Effect AC-0.0529 log-points.
* Interaction Effect (BC): Effect BC +0.0906 log-points.
* Interaction Effect (ABC): Effect $ABC=-0.0672~log-points.$

#### 5.2.2 Economic Interpretation

Analyzing the main effects, OverallQual (Factor B) emerges as the undisputed primary driver of property valuation (+0.4663), doing the bulk of the heavy lifting in our model's predictions. Notably, the isolated contributions of Central Air (+0.2878) and Exterior Quality (+0.2868) provide nearly identical standalone marginal increases.

The interaction effects, however, present a more nuanced market dynamic and prove that buyers do not evaluate these features in a vacuum:
* Threshold of Expectations $(AB=-0.1034)$: This significant negative interaction reveals an interesting reality regarding the diminishing marginal utility of equipment in the high-end segment. In premium homes, climate control is viewed as a baseline necessity rather than a luxury value-add, diminishing its relative premium. Conversely, introducing central air conditioning to a lower-tier property significantly spikes its baseline appeal, acting primarily as a necessary "correction" for standard goods to bring them up to market level.
* The Luxury Aesthetic Synergy $(BC=+0.0906)$: In contrast to comfort equipment, structural and finish qualities demonstrate a clear compounding effect. Fusing underlying structural quality (OverallQual) with a premium facade (ExterQual) creates a cohesive luxury aesthetic, generating a positive synergy that buyers are empirically willing to pay a distinct premium for.

For a developer, this observation serves as a strategic guide: when targeting the high-end, it is better to invest heavily in the synergy of structure and finishes (Factors B and C) rather than relying on standard comfort equipment. It proves, with statistical support, that the market is not merely an addition of characteristics, but a complex synergy that our future neural network must learn to model.

---

## 6 Parametric Regression

To model and explain the factors determining and impacting the sale price of real estate, we constructed a parametric model. This model expresses our target variable, $y=log(SalePrice),$ as a function of our significant variables, complemented by two pivotal dimensional indicators: habitable surface (GrLivArea) and basement surface (TotalBsmtSF).

### 6.1 Model Specification
We opted for an Ordinary Least Squares (OLS) specification:
$$y=\beta_{0}+\sum_{j=1}^{k}\beta_{j}x_{j}+\sum_{m=1}^{2}\gamma_{m}z_{m}+\epsilon$$
We encoded our ordinal variables $(x_{j})$ into numerical scales (e.g., $Po=1$ to $Ex=5$). To ensure model stability and avoid expanding the feature space unnecessarily, we treat these encoded ratings as discrete numeric variables, while utilizing Ridge and Lasso regularization to constrain potential multicollinearity. This formulation allows us to quantify separately what pertains to finish quality and what is purely due to the extension of built volumes.

### 6.2 Regression Results
Our OLS fit yields an adjusted $R^{2}$ of 0.815 (with a raw $R^{2}$ of 0.816), confirming that 81.5% of the variance in house prices is captured by these structural features.

**Table 6: OLS Model Coefficients**

| Feature | Coefficient | Std. Error | p-value |
|---|---|---|---|
| Intercept | 10.4001 | 0.047 | < 0.001 |
| OverallQual | 0.0883 | $0.006$ | <0.001 |
| ExterQual | 0.0491 | 0.013 | < 0.001 |
| BsmtQual | 0.0402 | $0.007$ | <0.001 |
| KitchenQual | 0.0739 | $0.010$ | <0.001 |
| FireplaceQu | 0.0199 | $0.003$ | <0.001 |
| LotShape | -0.0412 | 0.008 | < 0.001 |
| CentralAir | 0.2020 | $0.019$ | <0.001 |
| GrLivArea | 0.0002 | $0.00001$ | <0.001 |
| TotalBsmtSF | 0.0001 | $0.00001$ | <0.001 |

To validate the robustness of our model, we compared the OLS results with those using approaches such as Ridge $(R^{2}=0.816)$ and Lasso $(R^{2}=0.814)$. The observed stability of coefficients across these models indicates that our main predictors (Overall Qual and GrLivArea) are stable and not significantly biased by multicollinearity.

### 6.3 Model Diagnostics
Model validity is evaluated through the analysis of residual behavior and variance decomposition:
* Residual Analysis:
![Figure 4: Residual repartition around zero]()

We observe that the residual plot shows a homoscedastic distribution centered around 0, which reinforces the Gauss-Markov assumptions. Additionally, we verified the absence of heteroscedasticity, ensuring that the variance of the residuals remains constant across the range of predicted values, despite a minor presence of low-lying outliers. The absence of patterns, cycles, or repetitions confirms that our log-linear approach is appropriate.

* ANOVA on Regression Model: We performed a Type-II ANOVA to assess the partial contribution of each regressor. The significantly higher F-statistic for GrLiv Area reinforces that habitable surface remains the dominant driver of market value in our dataset, followed closely by overall quality.

**Table 7: Type-II ANOVA Summary (Main Predictors)**

| Source | Sum Sq | F | $PR(_{i}F)$ |
|---|---|---|---|
| GrLivArea | 11.12 | 376.3 | < 0.001 |
| OverallQual | 6.34 | 214.3 | < 0.001 |
| CentralAir | 3.20 | 108.2 | < 0.001 |
| TotalBsmtSF | 1.63 | 55.1 | < 0.001 |
| KitchenQual | 1.51 | 51.0 | < 0.001 |

The significantly higher F-statistic for GrLivArea reinforces that dimensional extension remains the dominant driver of market value in our dataset.

---

## 7 Neural-Network Regression

### 7.1 Model Inputs and Preprocessing

If we want to build a robust neural network it requires preprocessing workflow. Since neural networks don't have scale invariance, feeding raw values directly into the model would cause the gradient updates to become highly unstable. We implemented a comprehensive feature engineering and transformation pipeline to maximize the signal extracted from the dataset because we think that it is better than only relying on the existing features.

#### 7.1.1 Outlier Removal and Feature Engineering
Before creating new variables, we noticed a critical anomaly in the training set: a few massive houses with a living area (GrLivArea) exceeding 4000 square feet were selling for unusually low prices (under \$300,000). These extreme outliers would heavily skew the network's perception of price-per-square-foot, so we dropped them from the training data. Then, to help the neural network we directly created new features such as:
* Aggregated Dimensions: TotalSF is summing the basement, first-floor, and second-floor square footage (filling missing basement values with zero). Similarly, TotalBath was computed by logically weighting and combining full and half baths across all levels.
* Temporal Features: In general, Neural networks struggle with raw years. Instead of feeding in YearBuilt, we calculated the absolute HouseAge and RemodAge by subtracting the build and remodel years from the year the house was actually sold (YrSold).
* Interaction Terms: A large house is only valuable if it is well-built. I captured this synergistic effect by explicitly multiplying overall quality and condition (TotalQual), as well as overall quality by total square footage (Qual x_SF).
* Binarized Amenities: For features like pools, fireplaces, garages, and basements, the mere fact of having the amenity is more impactful than its exact size. If the house has the amenity, it "HasFeature" (e.g., HasPool, HasGarage) is 1, if not 0.

#### 7.1.2 Addressing Skewness and Target Transformation
In the first part of this project, we can clearly see that many features are right-skewed. To normalize these distributions and stabilize the network's learning process we take the logarithm (np.log1p) of these right-skewed features: LotArea, 1stFlrSF, GrLivArea, TotalSF, and TotalBsmtSF. We also apply this transformation to the target price since, we will train our model with logarithm of SalePrice.

#### 7.1.3 Imputation and Final Matrix Assembly
We split numerical and categorical features to apply different transformations:
* Numeric Features: Missing values were replaced by the median of each column to remain robust against any remaining outliers. In the same way, all numerical inputs were standardized using a StandardScaler (a mean of 0 and a variance of 1) to ensure stable weight initialization in the neural network and for efficiency reason.
* Categorical Features: Missing categorical values were replaced by the string "Missing" to treat absences as a distinct category. We then applied a One-Hot Encoder to expand these variables into a dense, binarized matrix.

After all these transformation, it is now the time to assemble the scaled numeric array and the encoded categorical array into a single horizontal stack (np.hstack).

### 7.2 Training and Hyperparameter Tuning

At first we tried to guess the best network architecture but we quickly find out it was impossible because there are a lot of parameters to tune, and their interactions are too complex to evaluate manually. To make life easier for us, we implemented Grid Search Cross-Validation (GridSearchCV) to systematically evaluate different configurations. We used Scikit-Learn's Multi-Layer Regressor MLPRegressor as our base estimator with relu for activation. For the optimization algorithm, we chose the lbfgs solver over the more common stochastic gradient descent methods like Adam. Adam is great for massive image datasets, lbfgs is a quasi-Newton method that often converges faster and yields better performance on smaller data sets. We also set arbitrarily an iteration limit (max_iter $=400($) to ensure the solver had a large space to work.

#### 7.2.1 The Hyperparameter Grid
The grid search was designed to balance model complexity against the risk of overfitting our 312 input features. I focused on two critical parameters:
* Hidden Layer Sizes: Since, our data set is small compared to typical big data we don't need highly complex networks. We tested three single-layer configurations (15, 30, and 40 neurons) to keep the model relatively shallow and one two-layer configuration (30 neurons followed by 15) to see if a slightly deeper architecture would improve our model.
* Regularization (Alpha): Overfitting is the model's biggest nightmare, especially when the number of features is larger than the number of rows. To avoid this, we tested four different values for the L2 penalty term (alpha: 1.0, 2.5, 5.0, and 7.5). A higher alpha heavily penalizes large weights, forcing the network to rely on other patterns rather than memorizing by heart.

#### 7.2.2 Cross-Validation and Final MLP
The grid search computed every combination using a 5-fold cross-validation which means we have (4 alpha x 4 set of hidden layers x 5 cross validation $=80$ combination). The scoring metric was set to neg root mean_squared_error. Because we applied a log transformation to our target variable (SalePrice), minimizing this RMSE effectively means the grid search is natively optimizing for the Root Mean Squared Logarithmic Error (RMSLE). This perfectly aligns our local training objective with Kaggle's scoring system.

The cross-validation results were highly revealing. The best performing model was surprisingly lightweight:
* Hidden Layer Size: A single hidden layer of just 15 neurons (hidden layer sizes=(15,)).
* Regularization: A very heavy L2 penalty of alpha $=7.5$.
* Local score Kaggle: 0.0936.

This optimal configuration confirms our hypothesis about small data: A shallow network forced to keep its weights exceptionally small (due to the high alpha) extracts only the most robust generalized pricing patterns.

### 7.3 Final Model Evaluation

We know that it is still possible to optimise certain parameters to reduce the RMSLE further. However, our initial goal was to get the RMSLE below 0.13. Having achieved that goal, we are satisfied. But now, it would be interesting to look at how the residual values behave.

![Figure 5: Plot for Residuals]()

Analyzing these visualizations in the logarithmic space perfectly illustrates the effectiveness of the transformation applied to our target variable. Looking at the residual scatter plot, the distribution is now much more homogeneous around the zero-error line. The variance of the errors remains remarkably constant, even for expensive properties. With a logarithmic Mean Absolute Error (MAE) of 0.0648 and an RMSE of 0.0936, the gap between these two indicator is now reduced. This proves that our model is very robust against outliers.

The histogram of the error distribution confirms what we've seen before. The curve has taken an pretty good Gaussian bell shape. The mean error is nearly 0.0000 which means that our model is free of systemic bias. It doesn't globally overvalue nor undervalues the real values.

To Summarize, these results prove that our feature engineering, the optimized regularization, and the shift to the logarithmic space has created a reliable and statistically good predicator.But we have to be careful because these plot are based on the trained data which means that the model had been already trained on it. This is reason why we have such a good score.

### 7.4 Kaggle Submission File

Since the model was trained on log-transformed prices, its raw outputs were still in logarithmic form. To convert these estimates back into US dollars, we applied the inverse exponential transformation (np.expm1). These final dollar values were then matched with their corresponding Id from the test dataset and exported into a submission.csv file. Our official Kaggle leaderboard score came out to be a strong 0.12689!

---

## References

[1] CStat26 Project Instructions.
https://github.com/lydiaYchen/CStat26/tree/main/Project

[2] Kaggle. House Prices: Advanced Regression Techniques.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
