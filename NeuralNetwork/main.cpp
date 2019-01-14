#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <ctime>
#include <random>
#include "Network.h"
#include "NetworkInfo.h"
#include "NetworkRead.h"

using namespace std;

int main(void)
{
	srand(unsigned(time(NULL)));

	NetworkInfo info;
	
	info.nLayer = 4;				//	输入+隐藏+输出层
	info.dimA = { 2, 10, 10, 1 };	//	每层神经元
	info.mu_init = 0.01;			//	Levenberg–Marquardt 算法的参数初始值
	info.nSample = 900;				//	样本个数
	info.ratio = 0.9;				//	训练集占比
	info.early_stop = 15;			//	early stopping 截止阈值

	string cmd;
	cin >> cmd;
	while (cmd != "exit") {

		if (cmd == "train") {
			Network network(info);		// 构造Network类

			ifstream fin("data.dat", ifstream::in);
			network.DataInput(fin);		// 读取数据
			fin.close();

			network.Train(cout);		// 训练

			ofstream fout("net.txt", ofstream::out);
			fout << setprecision(12) << scientific;
			network.OutputNet(fout);	// 输出神经网络参数
			fout.close();
		}
		else if (cmd == "read") {
			NetworkRead net(info);		// 构造NetworkRead类

			ifstream fin("net.txt", ifstream::in);
			net.Init(fin);				// NetworkRead类初始化
			fin.close();

			ofstream fout("test_net.dat", ofstream::out);
			fout << setprecision(12) << scientific;
			default_random_engine e(time(NULL));
			uniform_real_distribution<double> real(-3, 3);	// 随机数生成器，范围(-3, +3)

			Eigen::VectorXd xy(2);
			double z;
			for (int i = 0; i < 15000; ++i) {
				xy(0) = real(e);	// 随机生成坐标参数
				xy(1) = real(e);
				z = net(xy);		// 读取神经网络

				fout << setw(20) << left << xy(0);		// 输出结果到文件
				fout << setw(20) << left << xy(1);
				fout << setw(20) << left << z << endl;
			}
		}
		else
			cout << "invalid cmd!" << endl;

		cin >> cmd;
	}


	return 0;
}