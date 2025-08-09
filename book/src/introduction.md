# Introduction

This library provides a minimal C++ implementation of linear Gaussian state-space models (LGSSMs) and their associated algorithms, namely:

- Kalman filtering: estimating the state of a dynamical system given noisy sequential measurements.
- Rauch–Tung–Striebel (RTS) smoothing: refining the state estimates by incorporating future observations.
- Batch processing utilities for sequences of observations and controls.

## 1. Model formulation

We consider a linear dynamical system with additive Gaussian noise:

$$
x_k = A x_{k-1} + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)
$$
$$
z_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)
$$

where:

- $x_k \in \mathbb{R}^n$ is the state vector at time step $k$,
- $u_k \in \mathbb{R}^m$ is an optional control input,
- $z_k \in \mathbb{R}^p$ is the measurement vector,
- $A \in \mathbb{R}^{n \times n}$ is the state transition matrix,
- $B \in \mathbb{R}^{n \times m}$ is the control-input matrix,
- $H \in \mathbb{R}^{p \times n}$ is the observation matrix,
- $Q \in \mathbb{R}^{n \times n}$ is the process noise covariance,
- $R \in \mathbb{R}^{p \times p}$ is the measurement noise covariance.

## 2. Kalman filtering

The Kalman filter maintains the mean and covariance of the posterior distribution $ p(x_k \mid z_{1:k}) $ under the Gaussian assumption.

### Prediction step

Given the previous posterior $ (\hat{x}_{k-1}, P_{k-1}) $:

$$
\hat{x}^-_k = A \hat{x}_{k-1} + B u_k
$$
$$
P^-_k = A P_{k-1} A^\top + Q
$$

Here, $(\hat{x}^-_k, P^-_k)$ are the predicted state mean and covariance.

### Update step

With a new measurement $z_k$:

- Innovation (measurement residual):
$$
y_k = z_k - H \hat{x}^-_k
$$
- Innovation covariance:
$$
S_k = H P^-_k H^\top + R
$$
- Kalman gain:
$$
K_k = P^-_k H^\top S_k^{-1}
$$
- Updated mean and covariance:
$$
\hat{x}_k = \hat{x}^-_k + K_k y_k
$$
$$
P_k = (I - K_k H) P^-_k (I - K_k H)^\top + K_k R K_k^\top
$$

The filter proceeds recursively for each time step.

## 3. RTS smoothing

The Rauch–Tung–Striebel smoother refines the filtered estimates $ (\hat{x}_k, P_k) $ using backward recursions:

1. Initialize:
$$
\hat{x}^s_T = \hat{x}_T, \quad P^s_T = P_T
$$
2. For $k = T-1, \dots, 0$:
$$
C_k = P_k A^\top (P^-_{k+1})^{-1}
$$
$$
\hat{x}^s_k = \hat{x}_k + C_k \left( \hat{x}^s_{k+1} - \hat{x}^-_{k+1} \right)
$$
$$
P^s_k = P_k + C_k \left( P^s_{k+1} - P^-_{k+1} \right) C_k^\top
$$

This yields the posterior means/covariances given all observations $ z_{1:T} $.


## 4. What this library provides

This implementation:

- Encapsulates model parameters in a `KFConfig` struct.
- Provides a `KalmanFilter` class with:
  - Single-step `predict()` and `update()`.
  - Batch sequence filtering with `filter_sequence()`.
- Implements `rts_smooth_lti()` for RTS smoothing in the time-invariant case.
- Uses Eigen for matrix operations.
- Focuses on clarity and minimalism for teaching, prototyping, and integration into larger C++ projects.

## 5. References

- R.E. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, ASME J. Basic Eng., 1960.
- Rauch, H.E., Tung, F., and Striebel, C.T., *Maximum Likelihood Estimates of Linear Dynamic Systems*, AIAA Journal, 1965.

