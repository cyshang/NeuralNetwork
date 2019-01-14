#ifndef NETWORK_H_
#define NETWORK_H_

#include <Eigen/Dense>
#include <iostream>
#include <vector>

struct NetworkInfo;

class Network
{
public:
	Network(const NetworkInfo &);
	void DataInput(std::istream &);
	void Train(std::ostream &);
	void OutputNet(std::ostream &);

private:
	int nLayer;
	std::vector<int> dimA;
	int nSample;
	int tSample;
	int vSample;
	double mu_init;
	int early_stop;

	// 输入、隐藏、输出层
	std::vector<Eigen::MatrixXd> tA;
	std::vector<Eigen::MatrixXd> vA;

	// target值
	Eigen::RowVectorXd tE;
	Eigen::RowVectorXd vE;

	// 误差
	Eigen::RowVectorXd tErr;
	Eigen::RowVectorXd vErr;

	// RMSE
	double tRMSE;
	double tRMSE_last;
	double vRMSE;
	double vRMSE_last;	

	// 归一化相关
	Eigen::VectorXd maxX;
	Eigen::VectorXd avgX;
	Eigen::VectorXd minX;
	double maxE;
	double avgE;
	double minE;

	// 权重W和权重b
	std::vector<Eigen::MatrixXd> W;
	std::vector<Eigen::VectorXd> b;
	// 权重的备份
	std::vector<Eigen::MatrixXd> W_copy;
	std::vector<Eigen::VectorXd> b_copy;

	// 反向传播相关
	int nWeight;
	std::vector<Eigen::MatrixXd> dFdZ;
	Eigen::MatrixXd Jac;
	Eigen::MatrixXd JtJ;
	Eigen::VectorXd JtErr;
	Eigen::VectorXd dWeight;

	void ForwardProp(std::vector<Eigen::MatrixXd> &);
	void BackwardProp();
	bool CalcRMSE(const bool &);
	void InitWeight();
	void UpdateWeight(const double &);
	void BackupWeight();
	void RestoreWeight();
};

#endif // !NETWORK_H_
