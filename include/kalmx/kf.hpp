#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace kalmx {

/**
 * @brief Configuration parameters for a linear Kalman filter.
 *
 * This struct stores the system and noise matrices that define the
 * state-space model:
 *
 * xₖ = A xₖ₋₁ + B uₖ + wₖ,   wₖ ~ N(0, Q)
 * zₖ = H xₖ   + vₖ,          vₖ ~ N(0, R)
 */
struct KFConfig {
    /** @brief State transition matrix (A) */
    Eigen::MatrixXd A;

    /** @brief Control input matrix (B) */
    Eigen::MatrixXd B;

    /** @brief Observation matrix (H) */
    Eigen::MatrixXd H;

    /** @brief Process noise covariance (Q) */
    Eigen::MatrixXd Q;

    /** @brief Measurement noise covariance (R) */
    Eigen::MatrixXd R;

    /** @brief Initial state vector (x₀) */
    Eigen::VectorXd x0;

    /** @brief Initial error covariance matrix (P₀) */
    Eigen::MatrixXd P0;
};

/**
 * @brief Per-step statistics from a Kalman filter update.
 *
 * This struct stores the intermediate quantities computed during
 * the measurement update phase of the Kalman filter:
 *
 *   yₖ = zₖ − H xₖ⁻         (innovation or residual)
 *   Sₖ = H Pₖ⁻ Hᵀ + R       (innovation covariance)
 *
 * These can be useful for diagnostics, consistency checks,
 * and statistical testing (e.g., normalized innovation squared).
 */
struct StepStats {
    /**
     * @brief Innovation (measurement residual), yₖ.
     *
     * Difference between the actual measurement and the predicted
     * measurement from the current state estimate.
     */
    Eigen::VectorXd innovation;

    /**
     * @brief Innovation covariance matrix, Sₖ.
     *
     * Represents the expected covariance of the innovation, combining
     * state uncertainty and measurement noise.
     */
    Eigen::MatrixXd S;

    /**
     * @brief Construct a StepStats object.
     * @param innovation_ The innovation vector (yₖ).
     * @param covariance_ The innovation covariance matrix (Sₖ).
     */
    StepStats(Eigen::VectorXd innovation_, Eigen::MatrixXd covariance_)
        : innovation(std::move(innovation_)), S(std::move(covariance_)) {}
};

/**
 * @brief Discrete-time linear Kalman filter.
 *
 * Implements prediction and update steps for a linear Gaussian
 * state-space model:
 *
 * \f[
 *   x_k = A x_{k-1} + B u_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)
 * \f]
 * \f[
 *   z_k = H x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)
 * \f]
 *
 * Supports both step-by-step filtering and batch processing.
 */
class KalmanFilter {
  public:
    /**
     * @brief Construct a Kalman filter with given configuration.
     * @param cfg Configuration struct containing system matrices and initial state.
     */
    explicit KalmanFilter(const KFConfig &cfg);

    /**
     * @brief Predict the next state estimate.
     * @param u Optional control input vector \f$u_k\f$.
     *
     * If @p u is omitted or empty, the control term is skipped.
     * Updates the internal state mean @f$x_k^-@f$ and covariance @f$P_k^-@f$.
     */
    void predict(const Eigen::VectorXd &u = Eigen::VectorXd());

    /**
     * @brief Incorporate a new measurement to update the state.
     * @param z Measurement vector \f$z_k\f$.
     * @return Step statistics (innovation and innovation covariance).
     *
     * Computes the innovation \f$y_k\f$ and Kalman gain, then updates
     * the internal state mean @f$x_k@f$ and covariance @f$P_k@f$.
     */
    StepStats update(const Eigen::VectorXd &z);

    /**
     * @brief Access the current state estimate.
     * @return Constant reference to the state vector @f$x_k@f$.
     */
    [[nodiscard]] const Eigen::VectorXd &state() const { return x_; }

    /**
     * @brief Access the current state covariance.
     * @return Constant reference to the covariance matrix @f$P_k@f$.
     */
    [[nodiscard]] const Eigen::MatrixXd &cov() const { return P_; }

    /**
     * @brief Batch filtering output container.
     *
     * Stores filtered and predicted states/covariances at each time step.
     */
    struct BatchOut {
        /** @brief Filtered state means \f$x_k\f$ after each update. */
        std::vector<Eigen::VectorXd> x_filt;
        /** @brief Filtered covariances \f$P_k\f$ after each update. */
        std::vector<Eigen::MatrixXd> P_filt;
        /** @brief Predicted state means \f$x_k^-\f$ before each update. */
        std::vector<Eigen::VectorXd> x_pred;
        /** @brief Predicted covariances \f$P_k^-\f$ before each update. */
        std::vector<Eigen::MatrixXd> P_pred;
    };

    /**
     * @brief Filter an entire measurement sequence.
     * @param Z Sequence of measurements.
     * @param U Optional sequence of control inputs (same length as Z).
     * @return BatchOut containing per-step predicted and filtered estimates.
     *
     * Runs predict/update steps for each time step in the sequence.
     */
    BatchOut filter_sequence(const std::vector<Eigen::VectorXd> &Z,
                             const std::vector<Eigen::VectorXd> &U = {});

    /**
     * @brief State dimension @f$n\f$.
     */
    [[nodiscard]] int n() const { return static_cast<int>(x_.size()); }

    /**
     * @brief Control input dimension @f$m\f$.
     */
    [[nodiscard]] int m() const { return static_cast<int>(B_.cols()); }

    /**
     * @brief Measurement dimension @f$p\f$.
     */
    [[nodiscard]] int p() const { return static_cast<int>(H_.rows()); }

    /** @brief Access the state transition matrix @f$A@f$. */
    [[nodiscard]] const Eigen::MatrixXd &A() const { return A_; }
    /** @brief Access the control matrix @f$B@f$. */
    [[nodiscard]] const Eigen::MatrixXd &B() const { return B_; }
    /** @brief Access the observation matrix @f$H@f$. */
    [[nodiscard]] const Eigen::MatrixXd &H() const { return H_; }
    /** @brief Access the process noise covariance @f$Q@f$. */
    [[nodiscard]] const Eigen::MatrixXd &Q() const { return Q_; }
    /** @brief Access the measurement noise covariance @f$R@f$. */
    [[nodiscard]] const Eigen::MatrixXd &R() const { return R_; }

  private:
    Eigen::MatrixXd A_, B_, H_, Q_, R_;
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd I_; ///< Identity matrix of size n × n.
};

} // namespace kalmx
