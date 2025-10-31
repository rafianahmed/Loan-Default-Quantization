# Loan Default Quantization and Log-Likelihood Analysis

## Overview
This project analyzes loan data to estimate the probability of default (PD) based on borrowers’ FICO scores. The goal is to create a **rating map** that maps FICO scores to discrete credit ratings, where lower ratings indicate better creditworthiness. The approach uses **maximum likelihood estimation (MLE)** and **dynamic programming** to determine the optimal FICO score boundaries for a given number of buckets.

## Code Explanation

1. **Data Preparation**
```python
import pandas as pd
import os

# Set working directory and read CSV
os.chdir("path_to_csv_folder")
df = pd.read_csv('loan_data_created.csv')

# Extract relevant columns
x = df['default'].to_list()
y = df['fico_score'].to_list()
n = len(x)
x stores default status (0/1).

y stores FICO scores.

Arrays default and total track cumulative defaults and total borrowers per score.

Cumulative Count Calculation
default = [0 for i in range(851)]
total = [0 for i in range(851)]

for i in range(n):
    y[i] = int(y[i])
    default[y[i]-300] += x[i]
    total[y[i]-300] += 1

for i in range(0, 551):
    default[i] += default[i-1]
    total[i] += total[i-1]
Computes cumulative counts of defaults and total records for efficient calculation.
Log-Likelihood Function
import numpy as np

def log_likelihood(n, k):
    p = k/n
    if p == 0 or p == 1:
        return 0
    return k*np.log(p) + (n-k)*np.log(1-p)
Calculates log-likelihood of a bucket given n records and k defaults.

Dynamic Programming for Optimal Buckets
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
DP array stores max log-likelihood and previous index for each bucket.

Iteratively finds the optimal FICO score splits to maximize total log-likelihood.

Output Maximum Log-Likelihood and Boundaries
print(round(dp[r][550][0], 4))

k = 550
l = []
while r >= 0:
    l.append(k+300)
    k = dp[r][k][1]
    r -= 1

print(l)  # FICO score boundaries for buckets
Prints the maximum log-likelihood and the FICO score boundaries defining the buckets.

Usage

Place loan_data_created.csv in your working directory.

Update os.chdir() with your CSV folder path.

Adjust r to set the number of buckets.

Run the script to get optimal FICO boundaries and log-likelihood.

Dependencies

Python 3.x

pandas

numpy

Notes

This method generalizes to future datasets.

Useful for credit scoring, risk modeling, and discretizing continuous features while preserving predictive information.

FICO scores are assumed to range from 300 to 850.


