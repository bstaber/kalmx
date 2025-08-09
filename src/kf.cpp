#include "kalmx/kf.hpp"

namespace kalmx {

/**
 * @brief Check that a KFConfig struct has consistent matrix/vector dimensions.
 *
 * Verifies:
 *  - A is square (n×n)
 *  - B has n rows
 *  - H has n columns
 *  - Q is n×n
 *  - P0 is n×n
 *  - x0 has length n
 *  - R is square (p×p)
 *  - H has p rows (p = R.rows())
 *
 * @param config Kalman filter configuration to validate.
 *
 * @throw std::invalid_argument if any dimension check fails.
 */
static void check_dims(const KFConfig &config) {
    if (config.A.rows() != config.A.cols())
        throw std::invalid_argument("A must be square");
    const auto n_rows = config.A.rows();
    if (config.B.rows() != n_rows)
        throw std::invalid_argument("B rows != A rows");
    if (config.H.cols() != n_rows)
        throw std::invalid_argument("H cols != A rows");
    if (config.Q.rows() != n_rows || config.Q.cols() != n_rows)
        throw std::invalid_argument("Q must be A_rows x A_rows");
    if (config.P0.rows() != n_rows || config.P0.cols() != n_rows)
        throw std::invalid_argument("P0 must be A_rows x A_rows");
    if (config.x0.size() != n_rows)
        throw std::invalid_argument("x0 size != A rows");
    if (config.R.rows() != config.R.cols())
        throw std::invalid_argument("R must be square");
    if (config.H.rows() != config.R.rows())
        throw std::invalid_argument("H rows must equal R size");
}

/**
 * @brief Construct a Kalman filter with given configuration.
 *
 * Initializes all state matrices/vectors from @p cfg and checks
 * that dimensions are consistent.
 *
 * @param cfg Configuration struct containing model matrices and initial state.
 *
 * @throw std::invalid_argument if @p cfg has inconsistent dimensions.
 */
KalmanFilter::KalmanFilter(const KFConfig &cfg)
    : A_(cfg.A), B_(cfg.B), H_(cfg.H), Q_(cfg.Q), R_(cfg.R), x_(cfg.x0), P_(cfg.P0),
      I_(Eigen::MatrixXd::Identity(cfg.A.rows(), cfg.A.cols())) {
    check_dims(cfg);
}

/**
 * @brief Predict the next state mean/covariance.
 *
 * Uses the standard Kalman filter prediction step:
 * \f[
 *   x_{k|k-1} = A x_{k-1|k-1} \quad (+ B u_k \text{ if given})
 * \f]
 * \f[
 *   P_{k|k-1} = A P_{k-1|k-1} A^\top + Q
 * \f]
 *
 * @param u Control vector @f$u_k@f$ (optional). If empty, the control term is skipped.
 *
 * @throw std::invalid_argument if @p u is nonempty and has wrong size.
 */
void KalmanFilter::predict(const Eigen::VectorXd &u) {
    if (u.size() == 0) {
        x_ = A_ * x_;
    } else {
        if (u.size() != B_.cols())
            throw std::invalid_argument("u size != m");
        x_ = A_ * x_ + B_ * u;
    }
    P_ = A_ * P_ * A_.transpose() + Q_;
}

/**
 * @brief Update the state with a new observation.
 *
 * Implements the standard KF update equations:
 * \f[
 *   y_k = z_k - H x_{k|k-1}
 * \f]
 * \f[
 *   S_k = H P_{k|k-1} H^\top + R
 * \f]
 * \f[
 *   K_k = P_{k|k-1} H^\top S_k^{-1}
 * \f]
 * \f[
 *   x_{k|k} = x_{k|k-1} + K_k y_k
 * \f]
 * \f[
 *   P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^\top + K_k R K_k^\top
 * \f]
 *
 * @param z Observation vector @f$z_k@f$.
 * @return StepStats containing the innovation vector and innovation covariance.
 *
 * @throw std::invalid_argument if @p z has wrong size.
 */
StepStats KalmanFilter::update(const Eigen::VectorXd &z) {
    if (z.size() != H_.rows())
        throw std::invalid_argument("z size != p");
    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ = x_ + K * y;
    Eigen::MatrixXd IKH = I_ - K * H_;
    P_ = IKH * P_ * IKH.transpose() + K * R_ * K.transpose();
    return {y, S};
}

/**
 * @brief Run KF predict/update over a full sequence of observations.
 *
 * Optionally accepts a sequence of control vectors.
 *
 * @param Z Sequence of observations (length T).
 * @param U Sequence of controls (length T, optional; can be empty).
 * @return BatchOut struct containing all filtered and predicted states/covariances.
 *
 * @throw std::invalid_argument if any control/observation size is inconsistent.
 *
 * @note Complexity is O(T * n^3) due to matrix multiplies/inverses.
 */
KalmanFilter::BatchOut KalmanFilter::filter_sequence(const std::vector<Eigen::VectorXd> &Z,
                                                     const std::vector<Eigen::VectorXd> &U) {
    const std::size_t T = Z.size();
    BatchOut out;
    out.x_filt.reserve(T);
    out.P_filt.reserve(T);
    out.x_pred.reserve(T);
    out.P_pred.reserve(T);

    for (std::size_t k = 0; k < T; ++k) {
        Eigen::VectorXd xpred = A_ * x_;
        if (!U.empty()) {
            if (U[k].size() != B_.cols())
                throw std::invalid_argument("U[k] size != m");
            xpred += B_ * U[k];
        }
        out.x_pred.push_back(xpred);
        out.P_pred.push_back(A_ * P_ * A_.transpose() + Q_);

        if (U.empty()) {
            predict();
        } else {
            predict(U[k]);
        }
        update(Z[k]);
        out.x_filt.push_back(x_);
        out.P_filt.push_back(P_);
    }
    return out;
}

} // namespace kalmx
