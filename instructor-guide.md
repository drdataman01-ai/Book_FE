# Instructor Guide

*Financial Engineering in Practice*

This guide collects material that the printed book leaves out to save space: the learning objectives for every chapter and the open questions behind each chapter's research. Instructors can use it to plan a syllabus, set reading goals, write exam questions and choose topics for seminars or term papers. Each chapter of the book is organized around one question, and the chapter titles below are those questions. The capstone project, also in this repository, has its own objectives.

## Contents

- [Part 1. Learning objectives](#part-1-learning-objectives)
  - [Chapter 1. Does a Strong Track Record Prove Skill?](#chapter-1-does-a-strong-track-record-prove-skill)
  - [Chapter 2. Can Historical Data Show What a Strategy Would Have Earned?](#chapter-2-can-historical-data-show-what-a-strategy-would-have-earned)
  - [Chapter 3. Can Size and Value Explain Returns That Beta Cannot?](#chapter-3-can-size-and-value-explain-returns-that-beta-cannot)
  - [Chapter 4. Do Past Winners Keep Winning?](#chapter-4-do-past-winners-keep-winning)
  - [Chapter 5. Can Simple Trading Rules Beat Buy-and-Hold?](#chapter-5-can-simple-trading-rules-beat-buy-and-hold)
  - [Chapter 6. Can Valuations Predict the Equity Premium Out of Sample?](#chapter-6-can-valuations-predict-the-equity-premium-out-of-sample)
  - [Chapter 7. How Predictable Is Volatility?](#chapter-7-how-predictable-is-volatility)
  - [Chapter 8. How Reliably Do Value-at-Risk Models Measure Risk?](#chapter-8-how-reliably-do-value-at-risk-models-measure-risk)
  - [Chapter 9. Can Optimized Portfolios Beat an Equal Split Across Assets?](#chapter-9-can-optimized-portfolios-beat-an-equal-split-across-assets)
  - [Chapter 10. Does Diversification Fail When You Need It Most?](#chapter-10-does-diversification-fail-when-you-need-it-most)
  - [Chapter 11. Do High-Interest Currencies Depreciate Enough to Offset Their Yield?](#chapter-11-do-high-interest-currencies-depreciate-enough-to-offset-their-yield)
  - [Chapter 12. What Do Forward Rates Say About Future Interest Rates?](#chapter-12-what-do-forward-rates-say-about-future-interest-rates)
  - [Chapter 13. Do Investors Earn the Returns Their Stocks Earn?](#chapter-13-do-investors-earn-the-returns-their-stocks-earn)
  - [Chapter 14. Is a Profitable Backtest Profitable to Trade?](#chapter-14-is-a-profitable-backtest-profitable-to-trade)
  - [Chapter 15. Why Might Low-Risk Stocks Beat High-Risk Stocks?](#chapter-15-why-might-low-risk-stocks-beat-high-risk-stocks)
  - [Chapter 16. How Many Published Anomalies Hold Up?](#chapter-16-how-many-published-anomalies-hold-up)
  - [Chapter 17. What Do Credit Spreads Reflect?](#chapter-17-what-do-credit-spreads-reflect)
  - [Chapter 18. How Much Option Risk Does Delta Hedging Remove?](#chapter-18-how-much-option-risk-does-delta-hedging-remove)
  - [Chapter 19. Does Selling Options Earn a Reliable Premium?](#chapter-19-does-selling-options-earn-a-reliable-premium)
  - [Chapter 20. Can Machine Learning Predict Returns Better Than Linear Models?](#chapter-20-can-machine-learning-predict-returns-better-than-linear-models)
  - [Chapter 21. Can Foundation Models Forecast Better Than Classical Methods?](#chapter-21-can-foundation-models-forecast-better-than-classical-methods)
  - [Chapter 22. Can Transformers Price Stocks Better Than Factor Models?](#chapter-22-can-transformers-price-stocks-better-than-factor-models)
  - [Chapter 23. Can Foundation Models Trained on Market Data Beat GARCH?](#chapter-23-can-foundation-models-trained-on-market-data-beat-garch)
- [Part 2. Open questions for discussion](#part-2-open-questions-for-discussion)
  - [Chapter 4. Do Past Winners Keep Winning?](#chapter-4-do-past-winners-keep-winning-1)
  - [Chapter 5. Can Simple Trading Rules Beat Buy-and-Hold?](#chapter-5-can-simple-trading-rules-beat-buy-and-hold-1)
  - [Chapter 7. How Predictable Is Volatility?](#chapter-7-how-predictable-is-volatility-1)
  - [Chapter 8. How Reliably Do Value-at-Risk Models Measure Risk?](#chapter-8-how-reliably-do-value-at-risk-models-measure-risk-1)
  - [Chapter 9. Can Optimized Portfolios Beat an Equal Split Across Assets?](#chapter-9-can-optimized-portfolios-beat-an-equal-split-across-assets-1)
  - [Chapter 10. Does Diversification Fail When You Need It Most?](#chapter-10-does-diversification-fail-when-you-need-it-most-1)
  - [Chapter 11. Do High-Interest Currencies Depreciate Enough to Offset Their Yield?](#chapter-11-do-high-interest-currencies-depreciate-enough-to-offset-their-yield-1)
  - [Chapter 12. What Do Forward Rates Say About Future Interest Rates?](#chapter-12-what-do-forward-rates-say-about-future-interest-rates-1)
  - [Chapter 13. Do Investors Earn the Returns Their Stocks Earn?](#chapter-13-do-investors-earn-the-returns-their-stocks-earn-1)
  - [Chapter 15. Why Might Low-Risk Stocks Beat High-Risk Stocks?](#chapter-15-why-might-low-risk-stocks-beat-high-risk-stocks-1)
  - [Chapter 16. How Many Published Anomalies Hold Up?](#chapter-16-how-many-published-anomalies-hold-up-1)
  - [Chapter 17. What Do Credit Spreads Reflect?](#chapter-17-what-do-credit-spreads-reflect-1)
  - [Chapter 18. How Much Option Risk Does Delta Hedging Remove?](#chapter-18-how-much-option-risk-does-delta-hedging-remove-1)
  - [Chapter 19. Does Selling Options Earn a Reliable Premium?](#chapter-19-does-selling-options-earn-a-reliable-premium-1)
  - [Chapter 20. Can Machine Learning Predict Returns Better Than Linear Models?](#chapter-20-can-machine-learning-predict-returns-better-than-linear-models-1)
  - [Chapter 21. Can Foundation Models Forecast Better Than Classical Methods?](#chapter-21-can-foundation-models-forecast-better-than-classical-methods-1)
  - [Chapter 22. Can Transformers Price Stocks Better Than Factor Models?](#chapter-22-can-transformers-price-stocks-better-than-factor-models-1)
  - [Chapter 23. Can Foundation Models Trained on Market Data Beat GARCH?](#chapter-23-can-foundation-models-trained-on-market-data-beat-garch-1)

## Part 1. Learning objectives

Each list states what a student should be able to do after reading the chapter and completing its lab.

### Chapter 1. Does a Strong Track Record Prove Skill?

By the end of the chapter, students will be able to:

- compute simple, log and excess returns, and convert monthly means and volatilities to annual ones;
- calculate a Sharpe ratio and judge whether a mean return is statistically different from zero;
- distinguish profits that come from bearing risk, from luck and from skill, and measure skill as alpha;
- explain what market efficiency implies for return predictability, and what it does not;
- calculate how many years of data are needed to tell a good strategy from luck;
- list the main ways a backtest can mislead;
- explain how each chapter is organized around a question, read a chapter's summary of the evidence, and ask of any result: compared to what, out of sample, and after costs?

### Chapter 2. Can Historical Data Show What a Strategy Would Have Earned?

By the end of the chapter, students will be able to:

- explain how stock splits and dividends change quoted prices, and why research uses adjusted prices;
- compute market capitalization and distinguish value-weighted, equal-weighted and price-weighted portfolios;
- describe the main research datasets and their free counterparts, and explain delisting returns and point-in-time data;
- choose the right return measure for each task (simple or log, price or total, raw or excess), and explain why arithmetic and geometric averages differ;
- recognize survivorship bias, look-ahead bias, micro-cap effects and data snooping, and estimate how much each can distort a result;
- apply a data checklist before trusting any backtest.

### Chapter 3. Can Size and Value Explain Returns That Beta Cannot?

By the end of the chapter, students will be able to:

- summarize a return distribution by its mean, variance, skewness, kurtosis and autocorrelation, and run and interpret a linear regression;
- trace the path from the CAPM to the Fama–French three- and five-factor models and the factor zoo, and interpret beta, alpha and factor loadings at each step;
- split a portfolio's total variance into systematic and idiosyncratic parts;
- test whether factors are priced with the GRS test and Fama–MacBeth regressions;
- explain why searching through many factors produces false discoveries, and apply a stricter hurdle such as $t > 3$;
- separate statistical from economic significance, and judge what multi-factor models mean for asset management, performance attribution and corporate finance.

### Chapter 4. Do Past Winners Keep Winning?

By the end of the chapter, students will be able to:

- compute the standard 12–1 momentum signal, explain why it skips the most recent month, and distinguish cross-sectional from time-series momentum;
- explain momentum through underreaction and overreaction, and show how a long–short portfolio's beta changes over time;
- explain why momentum crashes after market rebounds, and manage that risk by scaling positions to a volatility target using only past data;
- examine the evidence on whether momentum still works over time, across markets and through its crashes.

### Chapter 5. Can Simple Trading Rules Beat Buy-and-Hold?

By the end of the chapter, students will be able to:

- define the moving-average and breakout rules and compute their signals;
- contrast time-series momentum with the cross-sectional momentum of Chapter 4;
- size positions to a volatility target using only past data;
- test the best of many rules with White's Reality Check and Hansen's SPA test;
- explain why trend following resembles owning a straddle;
- examine the evidence after publication, after costs and asset by asset.

### Chapter 6. Can Valuations Predict the Equity Premium Out of Sample?

By the end of the chapter, students will be able to:

- explain why in-sample $R^2$ flatters complex models, and why in-sample significance is a poor guide to out-of-sample performance;
- design a walk-forward evaluation using expanding or rolling windows with strict point-in-time information sets;
- recognize and prevent data leakage, applying purging and embargoing for overlapping targets and cross-validation;
- select natural hard-to-beat benchmarks and evaluate out-of-sample explanatory power ($R^2_{\text{OS}}$);
- execute hypothesis tests for forecast accuracy using Diebold–Mariano (for non-nested models) and Clark–West (for nested models);
- evaluate economic significance through certainty-equivalent return (CER) gains;
- apply White's (2000) Reality Check and Hansen's (2005) SPA test to evaluate data mining across multiple candidate forecasting models;
- analyze post-publication decay (McLean and Pontiff 2016) as a natural experiment;
- summarize the Goyal–Welch critique and evaluate the effectiveness of restricted and combined forecasts.

### Chapter 7. How Predictable Is Volatility?

By the end of the chapter, students will be able to:

- compute historical, EWMA, ARCH and GARCH volatility forecasts;
- explain realized variance, the HAR model, implied volatility and the VIX;
- evaluate forecasts with loss functions such as QLIKE and with the Diebold–Mariano test;
- judge whether volatility timing pays once forecasts are made in real time.

### Chapter 8. How Reliably Do Value-at-Risk Models Measure Risk?

By the end of the chapter, students will be able to:

- compute VaR and expected shortfall, and scale them across horizons;
- explain why VaR can penalize diversification;
- compare historical simulation, GARCH and filtered historical simulation VaR models;
- backtest a VaR model with the Kupiec and Christoffersen tests and the Basel traffic light, and judge a bank's risk disclosures.

### Chapter 9. Can Optimized Portfolios Beat an Equal Split Across Assets?

By the end of the chapter, students will be able to:

- compute the tangency and global minimum-variance portfolios, and explain why errors in estimated expected returns distort them most;
- describe the main repairs: constraints, shrinkage and risk budgeting;
- compare allocation rules out of sample with the Sharpe ratio, the certainty equivalent and the Jobson–Korkie test;
- judge whether an optimized portfolio beats equal weighting after costs, and summarize the evidence.

### Chapter 10. Does Diversification Fail When You Need It Most?

By the end of the chapter, students will be able to:

- explain why the average correlation limits diversification, and measure its benefit with the diversification ratio;
- correct crisis correlations for higher volatility with the Forbes–Rigobon method, and read tail correlations against a normal benchmark;
- explain why the stock–bond correlation changes sign with inflation;
- judge a portfolio by what it lost in bad months.

### Chapter 11. Do High-Interest Currencies Depreciate Enough to Offset Their Yield?

By the end of the chapter, students will be able to:

- compute a forward exchange rate from covered interest parity, and the profit or loss on a carry trade;
- run and interpret the Fama regression, and explain the forward premium puzzle;
- compare the explanations for the puzzle and the evidence for each;
- judge what a currency carry fund is paid for and how it loses money.

### Chapter 12. What Do Forward Rates Say About Future Interest Rates?

By the end of the chapter, students will be able to:

- compute zero-coupon yields, forward rates and the excess return on a bond, and use duration to measure its sensitivity to rates;
- explain the expectations hypothesis, the term premium, and the three classic tests of them;
- summarize the evidence, including doubts about how well the premium can be forecast out of sample;
- read a market-implied rate path as a forecast plus a premium, and judge a bond strategy built on carry and roll-down.

### Chapter 13. Do Investors Earn the Returns Their Stocks Earn?

By the end of the chapter, students will be able to:

- compute a holding-period return over a random horizon, explain what a stopping time is, and state what the optional stopping theorem says a selling rule can and cannot do;
- compare returns earned over holding periods of different lengths without being misled by annualizing short holds;
- explain the disposition effect through prospect theory and realization utility, and measure it with Odean's proportions of gains and losses realized;
- examine the evidence on whether an investor's selling decisions, taxes and the timing of money flows separate the return an asset earns from the return its holder receives.

### Chapter 14. Is a Profitable Backtest Profitable to Trade?

By the end of the chapter, students will be able to:

- read a limit order book and explain bid–ask spreads, market and limit orders, short selling and long–short portfolios;
- break trading costs into their parts, including market impact, and estimate the main ones from public data;
- compute a strategy's turnover, its net return and the break-even cost it can bear, and explain how market impact limits the money it can manage;
- design strategies that trade less without giving up much of their signal.

### Chapter 15. Why Might Low-Risk Stocks Beat High-Risk Stocks?

By the end of the chapter, students will be able to:

- estimate a beta, split total risk into systematic and idiosyncratic parts, and compute a CAPM alpha;
- draw the security market line, and explain how limits on leverage flatten it;
- build a beta-neutral betting-against-beta portfolio, and compare the explanations for the anomaly, such as leverage limits and lottery preferences;
- judge whether a minimum-volatility or BAB fund can deliver what its backtest promises.

### Chapter 16. How Many Published Anomalies Hold Up?

By the end of the chapter, students will be able to:

- explain market efficiency, the joint-hypothesis problem, and the kinds of replication, from reproduction to pure, scientific and out-of-sample replication;
- explain how publication bias inflates published results, and how empirical Bayes shrinkage corrects noisy estimates;
- set out the case that most anomalies fail and the case that most hold up, and the design choices that lead them to opposite conclusions;
- separate post-publication decay caused by selection from decay caused by investors' learning, and explain which standard of evidence suits which purpose.

### Chapter 17. What Do Credit Spreads Reflect?

By the end of the chapter, students will be able to:

- define default probabilities, recovery rates and loss given default, and compute the spread that expected losses justify;
- explain why the timing of defaults matters for the spread, and summarize the evidence on the credit spread puzzle;
- judge what a credit investor is paid for, and when reaching for yield is likely to disappoint.

### Chapter 18. How Much Option Risk Does Delta Hedging Remove?

By the end of the chapter, students will be able to:

- explain how Black–Scholes prices an option by replicating it, and which assumptions this needs;
- break a delta-hedged option's daily profit and loss into gamma and theta, and estimate the error from hedging only at intervals;
- explain why the volatility smile is evidence against the model's assumptions;
- judge what risks a “fully hedged” option seller or structured-product issuer still carries.

### Chapter 19. Does Selling Options Earn a Reliable Premium?

By the end of the chapter, students will be able to:

- use put–call parity to show that a covered call and a cash-secured put are the same position;
- define the VIX and the variance risk premium, and estimate an option seller's expected gain;
- judge an option strategy against a benchmark with the same market exposure, using up and down betas and a measure that cannot be gamed by selling crash insurance;
- judge what an option-income fund is likely to deliver, and what can go wrong.

### Chapter 20. Can Machine Learning Predict Returns Better Than Linear Models?

By the end of the chapter, students will be able to:

- explain how ridge regression, the lasso and the elastic net shrink coefficients to prevent overfitting;
- explain how a regression tree is grown, how random forests and gradient boosting combine many trees, and how a neural network computes and learns;
- choose hyperparameters in a walk-forward design, and read a model through feature importance, partial dependence and Shapley values;
- judge a machine-learning strategy by its value-weighted, net-of-cost performance against a linear model on the same inputs.

### Chapter 21. Can Foundation Models Forecast Better Than Classical Methods?

By the end of the chapter, students will be able to:

- name the classical benchmark for each kind of series, such as the historical mean, autoregressions, GARCH and HAR;
- explain how a foundation model is pretrained and used zero-shot, and why only data published after its training cutoff avoid contamination;
- compare forecasts on rolling origins with MSE ratios, the Diebold–Mariano test and the pinball and QLIKE losses, and work out how long a test must run to detect a small gain;
- judge the evidence on returns, volatility, other financial series and macroeconomic series.

### Chapter 22. Can Transformers Price Stocks Better Than Factor Models?

By the end of the chapter, students will be able to:

- explain the stochastic discount factor, and why pricing all assets correctly means finding the portfolio with the highest Sharpe ratio;
- describe how shrinkage, IPCA, deep networks and adversarial training have been used to estimate it;
- compute attention weights by hand, and explain how a transformer shares information across stocks;
- judge the evidence by out-of-sample pricing errors, Sharpe ratios, results by size group, and costs.

### Chapter 23. Can Foundation Models Trained on Market Data Beat GARCH?

By the end of the chapter, students will be able to:

- describe a price bar, and estimate daily volatility from its high, low, open and close;
- explain how Kronos turns price bars into tokens, and how a general model is fine-tuned without look-ahead;
- judge synthetic market data against the stylized facts of returns;
- compare forecasts with the Diebold–Mariano and Giacomini–White tests, using only data after each model's training cutoff.

## Part 2. Open questions for discussion

These questions mark where the research behind each chapter is still unsettled, so they have no single correct answer. They suit seminar discussions and term papers. A good answer states the competing explanations, names the evidence that would tell them apart, extends the chapter's lab to gather some of that evidence with free data, and reports what the published literature has found so far. Chapters 1, 2, 3, 6 and 14 have no open question.

### Chapter 4. Do Past Winners Keep Winning?

Can momentum crashes be forecast in real time, after the multiple-testing corrections of Chapter 16? And if crashes are the risk that the premium pays for, does managing them reduce the premium?

### Chapter 5. Can Simple Trading Rules Beat Buy-and-Hold?

Did the money that flowed into managed-futures funds after 2008 weaken trends, and will trend following profit in a faster or differently shaped crisis than 2008 and 2022?

### Chapter 7. How Predictable Is Volatility?

Does volatility timing pay after costs for factors such as momentum and in other countries, and can machine learning beat HAR under the out-of-sample discipline of Chapter 6?

### Chapter 8. How Reliably Do Value-at-Risk Models Measure Risk?

Can realized volatility from intraday returns (Chapter 7) catch the first day of a crisis, and how should regulators balance the accuracy of risk measures against stability?

### Chapter 9. Can Optimized Portfolios Beat an Equal Split Across Assets?

Do newer shrinkage methods make optimization over hundreds of stocks competitive with 1/N, and do machine-learning allocations avoid estimation error or hide it in more parameters (Chapter 20)?

### Chapter 10. Does Diversification Fail When You Need It Most?

Can the stock–bond correlation be forecast from inflation in time to adjust a portfolio, and how closely do private assets, whose smoothed valuations make them look like good diversifiers, really move with public stocks in a crisis?

### Chapter 11. Do High-Interest Currencies Depreciate Enough to Offset Their Yield?

If the premium is a slowly corrected mispricing, it should shrink as more money chases it; has it? And do momentum or volatility rules sidestep the worst months out of sample, or simply give up the premium?

### Chapter 12. What Do Forward Rates Say About Future Interest Rates?

Did central-bank bond buying after 2008 and in 2020 change how the term premium behaves, and does the Cochrane–Piazzesi factor hold up in other countries and after 2003?

### Chapter 13. Do Investors Earn the Returns Their Stocks Earn?

Is the disposition effect a preference, as realization utility says, or a belief, as Ben-David and Hirshleifer (2012) argue? And does it still drive momentum now that more stock is held by funds and algorithms that have no purchase price to anchor on?

### Chapter 15. Why Might Low-Risk Stocks Beat High-Risk Stocks?

Which risk measure is priced, beta, as leverage theories suggest, or volatility, as lottery theories suggest? And how large is the alpha of a value-weighted, beta-neutral portfolio of liquid stocks against the five-factor model?

### Chapter 16. How Many Published Anomalies Hold Up?

How many signals were tested but never published? The unknown size of this **file drawer** limits every multiple-testing correction. And is post-publication decay learning or selection, a question Part E answers in simulation but not yet in real data?

### Chapter 17. What Do Credit Spreads Reflect?

How much of the remainder is liquidity rather than risk, and how much of the premium is the price of dealers' limited capacity, given how sharply spreads fell when the Fed announced corporate bond purchases in 2020?

### Chapter 18. How Much Option Risk Does Delta Hedging Remove?

How much of the hedged seller's profit pays for jump risk, and do machine-learning “deep hedging” methods beat Black–Scholes out of sample and in crashes?

### Chapter 19. Does Selling Options Earn a Reliable Premium?

Options that expire on the day they are traded made up over 61% of S&P 500 index option volume in May 2025 (Cboe Global Markets 2025); have they changed the price of one-month volatility? And what do covered-call ETFs deliver after fees, spreads and taxes, compared with an index position with the same beta?

### Chapter 20. Can Machine Learning Predict Returns Better Than Linear Models?

How much of the edge remains among the largest 500 or 1,000 stocks, and is the virtue of complexity a general result or, as Nagel argues, disguised volatility-timed momentum? Chapters 21 to 23 ask the same question of transformers and time-series foundation models.

### Chapter 21. Can Foundation Models Forecast Better Than Classical Methods?

Will models pretrained on financial data beat classical benchmarks by enough to matter after costs, and can the field agree on post-cutoff benchmarks so that each new model is tested on data it cannot have seen?

### Chapter 22. Can Transformers Price Stocks Better Than Factor Models?

What does attention learn, given that observable links between firms explain only a small part of it? And if a Sharpe ratio above four reflects mispricing rather than risk, how long can it last once the model is widely used?

### Chapter 23. Can Foundation Models Trained on Market Data Beat GARCH?

How much accuracy on major indices comes from having seen them in pretraining, and can synthetic paths from generative models improve stress tests and capital estimates?
