#pragma once
#include <Eigen/Dense>
#include <vector>

namespace kalmx {

/**
 * @brief Output of the Rauch–Tung–Striebel (RTS) smoother.
 *
 * After running a forward Kalman filtering pass, the RTS smoother
 * performs a backward pass to refine the state estimates using
 * future information.
 *
 * - @c x_smooth[k]  : Smoothed state mean \f$\hat{x}_{k|T}\f$
 * - @c P_smooth[k]  : Smoothed state covariance \f$P_{k|T}\f$
 *
 * @note Sizes match the input filtering horizon T.
 */
struct RTSSmoothed {
    std::vector<Eigen::VectorXd> x_smooth; ///< \f$\hat{x}_{k|T}\f$, k = 0..T-1
    std::vector<Eigen::MatrixXd> P_smooth; ///< \f$P_{k|T}\f$,    k = 0..T-1
};

/**
 * @brief RTS smoother for linear time‑invariant (LTI) models.
 *
 * Assumes the forward pass has produced:
 *  - filtered means/covariances \f$\{\hat{x}_{k|k},\,P_{k|k}\}\f$
 *  - one‑step predictions \f$\{\hat{x}_{k|k-1},\,P_{k|k-1}\}\f$
 *
 * The backward recursion (for k = T-2..0):
 * \f[
 *   C_k = P_{k|k} A^\top \, P_{k+1|k}^{-1}
 * \f]
 * \f[
 *   \hat{x}_{k|T} = \hat{x}_{k|k} + C_k \big(\hat{x}_{k+1|T} - \hat{x}_{k+1|k}\big)
 * \f]
 * \f[
 *   P_{k|T} = P_{k|k} + C_k \big(P_{k+1|T} - P_{k+1|k}\big) C_k^\top
 * \f]
 *
 * @param A State transition matrix \f$A\f$ (n×n).
 * @param Q Process noise covariance \f$Q\f$ (n×n). (Not used explicitly here,
 *          but documented to mirror the LTI model; kept for potential square‑root
 *          or information‑form variants.)
 * @param x_filt Filtered means \f$\hat{x}_{k|k}\f$, k = 0..T-1 (size T).
 * @param P_filt Filtered covariances \f$P_{k|k}\f$, k = 0..T-1 (size T).
 * @param x_pred Predicted means \f$\hat{x}_{k|k-1}\f$, k = 0..T-1 (size T).
 * @param P_pred Predicted covariances \f$P_{k|k-1}\f$, k = 0..T-1 (size T).
 *
 * @return RTSSmoothed containing \f$\{\hat{x}_{k|T}, P_{k|T}\}\f$ for k = 0..T-1.
 *
 * @pre All vectors have the same length T; matrices/vectors have consistent
 *      dimensions (n). Each @p P_pred[k] must be invertible.
 *
 * @complexity \f$O(T\,n^3)\f$ due to the matrix inverse per step (use a solve
 *             with a factorization if you profile a bottleneck).
 *
 * @warning For numerical robustness, prefer solving
 *          \f$P_{k+1|k}^\top X^\top = A P_{k|k}^\top\f$ instead of explicit
 *          inversion, or use a square‑root smoother.
 */
[[nodiscard]] inline RTSSmoothed rts_smooth_lti(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Q,
                                                const std::vector<Eigen::VectorXd> &x_filt,
                                                const std::vector<Eigen::MatrixXd> &P_filt,
                                                const std::vector<Eigen::VectorXd> &x_pred,
                                                const std::vector<Eigen::MatrixXd> &P_pred) {
    const std::size_t T = x_filt.size();
    RTSSmoothed out;
    out.x_smooth.resize(T);
    out.P_smooth.resize(T);

    // Initialize with last filtered estimates
    out.x_smooth[T - 1] = x_filt[T - 1];
    out.P_smooth[T - 1] = P_filt[T - 1];

    // Backward pass
    for (std::ptrdiff_t k = static_cast<std::ptrdiff_t>(T) - 2; k >= 0; --k) {
        const Eigen::MatrixXd &Pf = P_filt[k];
        const Eigen::MatrixXd &Pp1 = P_pred[k + 1];

        // Smoother gain: C_k = P_{k|k} A^T (P_{k+1|k})^{-1}
        // (Consider replacing inverse with a solve for better stability.)
        Eigen::MatrixXd Ck = Pf * A.transpose() * Pp1.inverse();

        out.x_smooth[k] = x_filt[k] + Ck * (out.x_smooth[k + 1] - x_pred[k + 1]);
        out.P_smooth[k] = Pf + Ck * (out.P_smooth[k + 1] - Pp1) * Ck.transpose();
    }
    return out;
}

} // namespace kalmx
