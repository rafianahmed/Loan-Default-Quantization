
Loan Default Quantization and Log-Likelihood Analysis

Overview:
This script analyzes loan data to estimate the probability of default (PD) based on FICO scores.
It creates a rating map by splitting FICO scores into discrete buckets, where lower ratings
indicate better creditworthiness. The method uses maximum likelihood estimation (MLE) and
dynamic programming to determine optimal bucket boundaries.

Usage:
1. Place 'loan_data_created.csv' in your working directory.
2. Update the 'os.chdir()' line to point to your CSV folder.
3. Set 'r' to change the number of buckets.
4. Run the script to get optimal FICO boundaries and maximum log-likelihood.

Dependencies:
- Python 3.x
- pandas
- numpy
