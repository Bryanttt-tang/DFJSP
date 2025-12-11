#!/bin/bash

# Quick Setup Script for Hyperparameter Tuning
# Usage: bash setup_tuning.sh [optuna|skopt|both]

set -e

echo "========================================"
echo "Hyperparameter Tuning Setup"
echo "========================================"

# Parse arguments
MODE=${1:-optuna}

if [ "$MODE" = "optuna" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "Installing Optuna and dependencies..."
    pip install optuna optuna-dashboard plotly
    echo "✅ Optuna installed!"
fi

if [ "$MODE" = "skopt" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "Installing Scikit-Optimize..."
    pip install scikit-optimize
    echo "✅ Scikit-Optimize installed!"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Quick Start:"
echo ""

if [ "$MODE" = "optuna" ] || [ "$MODE" = "both" ]; then
    echo "1. Run Optuna tuning:"
    echo "   python tune_rule_based_hyperparameters.py"
    echo ""
    echo "2. View real-time dashboard (optional):"
    echo "   optuna-dashboard sqlite:///rule_based_tuning.db"
    echo ""
fi

if [ "$MODE" = "skopt" ] || [ "$MODE" = "both" ]; then
    echo "Run Scikit-Optimize tuning:"
    echo "   python tune_rule_based_skopt.py"
    echo ""
fi

echo "See HYPERPARAMETER_TUNING_GUIDE.md for details!"
echo ""
