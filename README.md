# MPC Path Tracking Simulation

This project implements a simple **Model Predictive Control (MPC)** simulation for vehicle path tracking.

The controller predicts future vehicle states and computes an optimal steering input while satisfying system constraints.

---

## Features

- Linear MPC controller
- Prediction horizon vs control horizon comparison
- Steering angle constraint
- Vehicle kinematic bicycle model
- Visualization using matplotlib animation

---

## MPC formulation

The controller solves the following optimization problem:

Minimize

J = Σ (xᵀQx + uᵀRu)

Subject to

x(k+1) = Ax(k) + Bu(k)

|δ| ≤ δ_max

---

## Parameters

Prediction horizon comparison

Vehicle A  
Np = 3  

Vehicle B  
Np = 15  

Control horizon  

Nc = 2

---

## Dependencies
numpy
matplotlib
cvxpy


---

## Run

```bash
python true_mpc_horizon_compare.py