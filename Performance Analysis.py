#!/usr/bin/env python
# coding: utf-8

# In[27]:


"Dillithium Codes"
import psutil
import time
import os
from ctypes import *
from ctypes.util import find_library

# Load the PQCLEAN library
pqclean = CDLL(find_library("pqclean_dilithium"))

start_time = time.time()
# Set the custom ring dimension
n = 64

# Set the parameter constants based on the ring dimension
PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES = n//8 + 2*n*1216//8 + 2*32
PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES = n*1152//8 + 32

# Allocate memory for the public and secret keys
pk = create_string_buffer(PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES)
sk = create_string_buffer(PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES)

# Generate the key pair
pqclean.crypto_sign_keypair(pk, sk)

end_time = time.time()
# Print the keys
print("Public key:", pk.raw.hex())
print("Secret key:", sk.raw.hex())



cpu_percent = psutil.cpu_percent(interval=None, percpu=False)
cpu_time = (end_time - start_time) * cpu_percent / 100

print("CPU Usage: {:.2f}%".format(cpu_percent))
print("CPU Time: {:.2f}s".format(cpu_time))
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
process = psutil.Process()
memory_info = process.memory_info()
print(f"Current RAM usage: {memory_info.rss / 1024 / 1024} MB")


# In[81]:


"Falcon Codes"
import psutil
import time
import os
from ctypes import *
from ctypes.util import find_library
start_time = time.time()
import falcon
import random
from math import log2, ceil
start_time = time.time()
# FALCON parameters
n = 65  # Ring dimension
q = 12289  # Modulus
p = 512  # Polynomial degree
d = 3  # Number of degree-1 coefficients

# Step 1: Generate private key
F = [random.randint(0, q - 1) for _ in range(n)]
G = [random.randint(0, q - 1) for _ in range(n)]
F_inv = [pow(f, -1, q) for f in F]
sk = [random.randint(0, q - 1) for _ in range(n)]

# Step 2: Compute public key
A = [[0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        A[i][j] = (F[i] * G[j] - sk[i] * F_inv[j]) % q
        pk = A
end_time = time.time()
print (pk)
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# In[91]:


"Dillithium Codes Final"
import time
import falcon
import numpy as np
import pandas as pd

# Define a list of values for "n"
n_values = []
for i in range(200):
    n_values.append(random.randint(100, 500)) 

# Initialize an empty dictionary to store elapsed times for each "n" value
elapsed_times = {}

# Iterate through each "n" value
for n in n_values:
    start_time = time.time()  # Record start time for current iteration
# Set the parameter constants based on the ring dimension
    PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES = n//8 + 2*n*1216//8 + 2*32
    PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES = n*1152//8 + 32

# Allocate memory for the public and secret keys
    pk = create_string_buffer(PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_PUBLICKEYBYTES)
    sk = create_string_buffer(PQCLEAN_DILITHIUM2_CLEAN_CRYPTO_SECRETKEYBYTES)

    end_time = time.time()  # Record end time for current iteration
    elapsed_time = end_time - start_time  # Calculate elapsed time for current iteration
    elapsed_times[n] = elapsed_time  # Store elapsed time in dictionary with "n" as key

# Print elapsed times for each "n" value
for n, elapsed_time in elapsed_times.items():
    print(f"Elapsed time for n = {n}: {elapsed_time} seconds")
    
df = pd.DataFrame(list(elapsed_times.items()), columns=['n', 'elapsed_time'])
print(df)

df.to_excel('/Users/Maryam/Desktop/elapsed_times.xlsx', index=False)


# In[92]:


"Falcon Codes Final"
import random
import math
import time

q = 12289  # Modulus
p = 512  # Polynomial degree
d = 3  # Number of degree-1 coefficients

# Define a list of values for "n"
n_values = []
for i in range(200):
    n_values.append(random.randint(100, 500))

# Initialize an empty dictionary to store elapsed times for each "n" value
elapsed_times = {}

# Iterate through each "n" value
for n in n_values:
    start_time = time.time()  # Record start time for current iteration
    F = [random.randint(0, q - 1) for _ in range(n)]
    G = [random.randint(0, q - 1) for _ in range(n)]
    F_inv = [0] * n  # Initialize F_inv with zeros
    for i in range(n):
        # Check if F[i] is coprime to q
        if math.gcd(F[i], q) == 1:
            F_inv[i] = pow(F[i], -1, q)
        else:
            # Handle case where F[i] is not invertible
            print(f"Warning: {F[i]} is not coprime to {q}. Skipping.")
            continue

    sk = [random.randint(0, q - 1) for _ in range(n)]
    A = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = (F[i] * G[j] - sk[i] * F_inv[j]) % q
            pk = A

    end_time = time.time()  # Record end time for current iteration
    elapsed_time = end_time - start_time  # Calculate elapsed time for current iteration
    elapsed_times[n] = elapsed_time  # Store elapsed time in dictionary with "n" as key

# Print elapsed times for each "n" value
for n, elapsed_time in elapsed_times.items():
    print(f"Elapsed time for n = {n}: {elapsed_time} seconds")
    
df = pd.DataFrame(list(elapsed_times.items()), columns=['n', 'elapsed_time'])
print(df)

df.to_excel('/Users/Maryam/Desktop/elapsed_times2.xlsx', index=False)


# In[102]:


"Dilithium OLS"
import pandas as pd
import statsmodels.api as sm

# Specify the correct file path
file_path = '/Users/Maryam/Desktop/Maryam/elapsed_times.xlsx'

# Read the Excel file
df = pd.read_excel(file_path, sheet_name=0)
# Extract the predictor variables (X) and the response variable (y)
X = df[['n']]  # replace with actual column names
y = df['elapsed_time']  # replace with actual column name

# Add a constant to the predictor variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit(cov_type='HC1') 

# Print the regression results
print(results.summary())


# In[101]:


"FALCON OLS"
import pandas as pd
import statsmodels.api as sm

# Specify the correct file path
file_path = '/Users/Maryam/Desktop/Maryam/elapsed_times2.xlsx'

# Read the Excel file
df = pd.read_excel(file_path, sheet_name=0)
# Extract the predictor variables (X) and the response variable (y)
X = df[['n']]  # replace with actual column names
y = df['elapsed_time']  # replace with actual column name

# Add a constant to the predictor variables
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit(cov_type='HC1') 

# Print the regression results
print(results.summary())

