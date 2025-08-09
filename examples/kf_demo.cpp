#include "kalmx/kf.hpp"
#include "kalmx/rts.hpp"
#include <iostream>
#include <random>

int main()
{
  const double dt = 0.1;
  Eigen::Matrix2d A; A << 1.0, dt, 0.0, 1.0;
  Eigen::Matrix<double,2,1> B; B << 0.0, 0.0;
  Eigen::RowVector2d H; H << 1.0, 0.0;

  Eigen::Matrix2d Q = (Eigen::Matrix2d() << 1e-4, 0, 0, 1e-4).finished();
  Eigen::MatrixXd R(1,1); R(0,0) = 1e-2;

  Eigen::Vector2d x0(0.0, 1.0);
  Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

  kalmx::KFConfig cfg{A, B, H, Q, R, x0, P0};
  kalmx::KalmanFilter kf(cfg);

  std::mt19937 rng(42);
  std::normal_distribution<double> w(0.0, std::sqrt(R(0,0)));

  const int T = 100;
  std::vector<Eigen::VectorXd> Z; Z.reserve(T);
  std::vector<Eigen::VectorXd> U;

  double pos = 0.0, vel = 1.0;
  for (int k = 0; k < T; ++k) {
    pos += dt * vel;
    Eigen::VectorXd z(1); z(0) = pos + w(rng);
    Z.push_back(z);
  }

  auto out = kf.filter_sequence(Z, U);
  auto sm = kalmx::rts_smooth_lti(A, Q, out.x_filt, out.P_filt, out.x_pred, out.P_pred);

  std::cout << "Filtered final state: " << out.x_filt.back().transpose() << "\n";
  std::cout << "Smoothed  final state: " << sm.x_smooth.front().transpose() << "\n";
  return 0;
}
