#include <fstream>
#include <iomanip>
#include "NetworkInfo.h"
#include "NetworkRead.h"

using namespace Eigen;
using namespace std;

NetworkRead::NetworkRead(const NetworkInfo & info)
{
	nLayer = info.nLayer;
	dimA = info.dimA;

	A.resize(nLayer);
	for (int i = 0; i < nLayer; ++i) {
		A.resize(dimA[i]);
	}

	W.resize(nLayer);
	b.resize(nLayer);
	for (int i = 1; i < nLayer; ++i) {
		W[i].resize(dimA[i], dimA[i - 1]);
		b[i].resize(dimA[i]);
	}

	avgX.resize(dimA[0]);
	rangeX.resize(dimA[0]);
}

void NetworkRead::Init(std::istream & fin)
{
	int dimX = dimA[0];
	VectorXd minX(dimX), maxX(dimX);
	double minE, maxE;

	for (int i = 0; i < dimX; ++i) {
		fin >> minX(i);
	}

	for (int i = 0; i < dimX; ++i) {
		fin >> avgX(i);
	}

	for (int i = 0; i < dimX; ++i) {
		fin >> maxX(i);
	}

	rangeX = maxX - minX;

	fin >> minE >> avgE >> maxE;

	rangeE = maxE - minE;

	for (int iLayer = 1; iLayer < nLayer; ++iLayer) {
		double *ptr;

		for (ptr = W[iLayer].data(); ptr != W[iLayer].data() + W[iLayer].size(); ++ptr) {
			fin >> *ptr;
		}

		for (ptr = b[iLayer].data(); ptr != b[iLayer].data() + b[iLayer].size(); ++ptr) {
			fin >> *ptr;
		}
	}

	ofstream fout;
	fout.open("debug.txt", ofstream::out);
	OutputNet(fout);
	fout.close();
}

void NetworkRead::OutputNet(ostream & fout)
{

	for (int iLayer = 1; iLayer < nLayer; ++iLayer) {
		Map<RowVectorXd> mapW(W[iLayer].data(), W[iLayer].size());
		Map<RowVectorXd> mapb(b[iLayer].data(), b[iLayer].size());
		fout << mapW << endl;
		fout << mapb << endl;
	}
}