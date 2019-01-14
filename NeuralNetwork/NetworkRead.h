#ifndef NETWORKREAD_H_
#define NETWORKREAD_H_

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

struct NetworkInfo;

class NetworkRead
{
public:
	NetworkRead(const NetworkInfo &);
	void Init(std::istream &);
	template <typename Derived>
	double operator()(const Eigen::MatrixBase<Derived> &);

private:
	int nLayer;
	std::vector<int> dimA;

	std::vector<Eigen::VectorXd> A;

	Eigen::VectorXd avgX;
	Eigen::VectorXd rangeX;
	double avgE;
	double rangeE;

	std::vector<Eigen::MatrixXd> W;
	std::vector<Eigen::VectorXd> b;

	void OutputNet(std::ostream &);
};

template <typename Derived>
double NetworkRead::operator()(const Eigen::MatrixBase<Derived> & x)
{
	A[0] = (x - avgX).array() / rangeX.array();

	for (int i = 1; i < nLayer - 1; ++i) {
		A[i] = ((W[i] * A[i - 1]).colwise() + b[i]).array() /
			sqrt(((W[i] * A[i - 1]).colwise() + b[i]).array().square() + 1);
	}

	A[nLayer - 1] = (W[nLayer - 1] * A[nLayer - 2]).colwise() + b[nLayer - 1];

	return A[nLayer - 1](0) * rangeE + avgE;
}

#endif // !NETWORKREAD_H_

