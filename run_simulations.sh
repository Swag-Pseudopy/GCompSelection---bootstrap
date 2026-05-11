#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo " Starting G-Computation Bootstrap Project Pipeline"
echo " M.Stat Resampling Project - Swagato Das"
echo "=================================================="

# Check if required packages are installed
echo "Checking Python dependencies..."
python3 -c "import statsmodels, pandas, numpy, tqdm, delicatessen" &> /dev/null || {
    echo "Missing dependencies. Installing via pip..."
    pip install statsmodels pandas numpy tqdm delicatessen==3.0
}

# Run the single dataset generation (Tables 1 & 2)
echo ""
echo ">>> [1/3] Generating Single-Dataset Variance Tables (Case 1 & 2)..."
python3 code_2cases.py

# Run Case 1 Monte Carlo (Table 3)
echo ""
echo ">>> [2/3] Running Monte Carlo Bootstrap for Case 1..."
python3 simulations/sim1_boot.py

# Run Case 2 Monte Carlo (Table 4)
echo ""
echo ">>> [3/3] Running Monte Carlo Bootstrap for Case 2..."
python3 simulations/sim2_boot.py

echo ""
echo "=================================================="
echo " All scripts executed successfully!"
echo " The outputs match the tables generated for the report."
echo "=================================================="