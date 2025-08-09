#pragma once
#include <Eigen/Dense>
#include <vector>

namespace kalmx {

struct RTSSmoothed {
    std::vector<Eigen::VectorXd> x_smooth;
    std::vector<Eigen::MatrixXd> P_smooth;
};

inline RTSSmoothed rts_smooth_lti(const Eigen::MatrixXd &A, const Eigen::MatrixXd &Q,
                                  const std::vector<Eigen::VectorXd> &x_filt,
                                  const std::vector<Eigen::MatrixXd> &P_filt,
                                  const std::vector<Eigen::VectorXd> &x_pred,
                                  const std::vector<Eigen::MatrixXd> &P_pred) {
    const std::size_t T = x_filt.size();
    RTSSmoothed out;
    out.x_smooth.resize(T);
    out.P_smooth.resize(T);
    out.x_smooth[T - 1] = x_filt[T - 1];
    out.P_smooth[T - 1] = P_filt[T - 1];

    for (std::ptrdiff_t k = static_cast<std::ptrdiff_t>(T) - 2; k >= 0; --k) {
        const Eigen::MatrixXd &Pf = P_filt[k];
        const Eigen::MatrixXd &Pp1 = P_pred[k + 1];
        Eigen::MatrixXd Ck = Pf * A.transpose() * Pp1.inverse();
        out.x_smooth[k] = x_filt[k] + Ck * (out.x_smooth[k + 1] - x_pred[k + 1]);
        out.P_smooth[k] = Pf + Ck * (out.P_smooth[k + 1] - Pp1) * Ck.transpose();
    }
    return out;
}

} // namespace kalmx
