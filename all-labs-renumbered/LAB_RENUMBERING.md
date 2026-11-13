# Lab renumbering (November 2026)

The book has 24 chapters. The old Chapter 4 (data mining) is dropped, the out-of-sample
and machine-learning chapters are merged into Chapter 4 (which uses the machine-learning
lab), and the portfolio chapter moves up to Chapter 6, right after Costs. Every lab
carries its printed chapter number.

| Original folder | New folder |
|---|---|
| ch01-brilliant-by-chance, ch02-data-traps, ch03-noisy-returns | unchanged |
| ch04-overfitting | retired |
| ch05-forecasting | retired (its walk-forward code remains in fintest/forecast.py) |
| ch21-ml | ch04-ml |
| ch06-costs | ch05-costs |
| ch19-portfolios | ch06-portfolios |
| ch07-predictability ... ch18-carry | unchanged numbers (ch07 ... ch18) |
| ch20-diversification | ch19-diversification |
| ch22-foundation, ch23-news, ch24-investors | ch20-foundation, ch21-news, ch22-investors |
| ch25-design, ch26-capstone | ch23-design, ch24-capstone |

Inside every lab the same mapping was applied to "Lab N", "Chapter N", "Section N.k",
script, test and notebook names (labNN.py, labNN_walkthrough.ipynb, tests/test_labNN.py),
report names and figure files (figN_k_*.png). Lab figures are captioned "Lab Figure N.k"
so they cannot be confused with the book's own Figure N.k. All 24 test suites pass.
