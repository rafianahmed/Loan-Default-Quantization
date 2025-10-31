"""
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
"""

import pandas as pd
import numpy as np
import os

# --- Data Preparation ---
# Set working directory to the folder containing CSV
os.chdir("path_to_csv_folder")  # <-- replace with your path

# Read loan data
df = pd.read_csv('loan_data_created.csv')

# Extract default status and FICO scores
x = df['default'].to_list()      # 0/1 for default
y = df['fico_score'].to_list()   # FICO scores
n = len(x)

# Initialize cumulative arrays
default = [0 for i in range(851)]  # cumulative defaults
total = [0 for i in range(851)]    # cumulative totals

# --- Cumulative Count Calculation ---
for i in range(n):
    y[i] = int(y[i])
    default[y[i]-300] += x[i]
    total[y[i]-300] += 1

for i in range(0, 551):
    default[i] += default[i-1]
    total[i] += total[i-1]

# --- Log-Likelihood Function ---
def log_likelihood(n, k):
    """
    Computes the log-likelihood for a bucket of n records with k defaults.
    """
    p = k/n
    if p == 0 or p == 1:
        return 0
    return k*np.log(p) + (n-k)*np.log(1-p)

# --- Dynamic Programming for Optimal Buckets ---
r = 10  # number of buckets
dp = [[[-10**18, 0] for i in range(551)] for j in range(r+1)]

for i in range(r+1):
    for j in range(551):
        if i == 0:
            dp[i][j][0] = 0
        else:
            for k in range(j):
                if total[j] == total[k]:
                    continue
                if i == 1:
                    dp[i][j][0] = log_likelihood(total[j], default[j])
                else:
                    candidate = dp[i-1][k][0] + log_likelihood(total[j]-total[k], default[j]-default[k])
                    if dp[i][j][0] < candidate:
                        dp[i][j][0] = candidate
                        dp[i][j][1] = k

# --- Output Results ---
print("Maximum Log-Likelihood:", round(dp[r][550][0], 4))

# Trace back to get FICO boundaries
k = 550
l = []
while r >= 0:
    l.append(k+300)
    k = dp[r][k][1]
    r -= 1

print("Optimal FICO Score Boundaries for Buckets:", l)

"""
Explanation:
1. Data is read from a CSV with 'default' and 'fico_score' columns.
2. Cumulative defaults and totals are calculated for each FICO score.
3. log_likelihood(n,k) calculates the probability of defaults in a bucket.
4. Dynamic programming (DP) finds the optimal bucket boundaries that maximize total log-likelihood.
5. Outputs the maximum log-likelihood value and the corresponding FICO score boundaries.
6. This method generalizes to future datasets, useful for credit scoring, risk modeling, and discretizing continuous variables.
"""
