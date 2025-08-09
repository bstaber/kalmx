#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

namespace kalmx {

struct KFConfig {
    Eigen::MatrixXd A, B, H, Q, R;
    Eigen::VectorXd x0;
    Eigen::MatrixXd P0;
};

struct StepStats {
    Eigen::VectorXd innovation;
    Eigen::MatrixXd S;
};

class KalmanFilter {
  public:
    explicit KalmanFilter(const KFConfig &cfg);
    void predict(const Eigen::VectorXd &u = Eigen::VectorXd());
    StepStats update(const Eigen::VectorXd &z);
    const Eigen::VectorXd &state() const { return x_; }
    const Eigen::MatrixXd &cov() const { return P_; }

    struct BatchOut {
        std::vector<Eigen::VectorXd> x_filt;
        std::vector<Eigen::MatrixXd> P_filt;
        std::vector<Eigen::VectorXd> x_pred;
        std::vector<Eigen::MatrixXd> P_pred;
    };
    BatchOut filter_sequence(const std::vector<Eigen::VectorXd> &Z,
                             const std::vector<Eigen::VectorXd> &U = {});

    int n() const { return static_cast<int>(x_.size()); }
    int m() const { return static_cast<int>(B_.cols()); }
    int p() const { return static_cast<int>(H_.rows()); }

    const Eigen::MatrixXd &A() const { return A_; }
    const Eigen::MatrixXd &B() const { return B_; }
    const Eigen::MatrixXd &H() const { return H_; }
    const Eigen::MatrixXd &Q() const { return Q_; }
    const Eigen::MatrixXd &R() const { return R_; }

  private:
    Eigen::MatrixXd A_, B_, H_, Q_, R_;
    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd I_;
};

} // namespace kalmx
