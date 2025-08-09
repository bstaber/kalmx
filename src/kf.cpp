#include "kalmx/kf.hpp"

namespace kalmx {

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
        throw std::invalid_argument("x0 size != Açrows");
    if (config.R.rows() != config.R.cols())
        throw std::invalid_argument("R must be square");
    if (config.H.rows() != config.R.rows())
        throw std::invalid_argument("H rows must equal R size");
}

KalmanFilter::KalmanFilter(const KFConfig &cfg)
    : A_(cfg.A), B_(cfg.B), H_(cfg.H), Q_(cfg.Q), R_(cfg.R), x_(cfg.x0), P_(cfg.P0),
      I_(Eigen::MatrixXd::Identity(cfg.A.rows(), cfg.A.cols())) {
    check_dims(cfg);
}

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
