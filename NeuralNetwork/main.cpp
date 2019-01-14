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
	
	info.nLayer = 4;				//	����+����+�����
	info.dimA = { 2, 10, 10, 1 };	//	ÿ����Ԫ
	info.mu_init = 0.01;			//	Levenberg�CMarquardt �㷨�Ĳ�����ʼֵ
	info.nSample = 900;				//	��������
	info.ratio = 0.9;				//	ѵ����ռ��
	info.early_stop = 15;			//	early stopping ��ֹ��ֵ

	string cmd;
	cin >> cmd;
	while (cmd != "exit") {

		if (cmd == "train") {
			Network network(info);		// ����Network��

			ifstream fin("data.dat", ifstream::in);
			network.DataInput(fin);		// ��ȡ����
			fin.close();

			network.Train(cout);		// ѵ��

			ofstream fout("net.txt", ofstream::out);
			fout << setprecision(12) << scientific;
			network.OutputNet(fout);	// ������������
			fout.close();
		}
		else if (cmd == "read") {
			NetworkRead net(info);		// ����NetworkRead��

			ifstream fin("net.txt", ifstream::in);
			net.Init(fin);				// NetworkRead���ʼ��
			fin.close();

			ofstream fout("test_net.dat", ofstream::out);
			fout << setprecision(12) << scientific;
			default_random_engine e(time(NULL));
			uniform_real_distribution<double> real(-3, 3);	// ���������������Χ(-3, +3)

			Eigen::VectorXd xy(2);
			double z;
			for (int i = 0; i < 15000; ++i) {
				xy(0) = real(e);	// ��������������
				xy(1) = real(e);
				z = net(xy);		// ��ȡ������

				fout << setw(20) << left << xy(0);		// ���������ļ�
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