#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "Network.h"
#include "NetworkInfo.h"

using namespace std;
using namespace Eigen;

Network::Network(const NetworkInfo & info)	
{
	nLayer = info.nLayer;
	dimA = info.dimA;
	nSample = info.nSample;
	tSample = static_cast<int>(nSample * info.ratio);
	vSample = nSample - tSample;
	mu_init = info.mu_init;
	early_stop = info.early_stop;

	tA.resize(nLayer);
	vA.resize(nLayer);
	for (int i = 0; i < nLayer; ++i) {
		tA[i].resize(dimA[i], tSample);
		vA[i].resize(dimA[i], vSample);		
	}	

	tE.resize(tSample);
	vE.resize(vSample);

	tErr.resize(tSample);
	vErr.resize(vSample);

	maxX.resize(dimA[0]);
	avgX.resize(dimA[0]);
	minX.resize(dimA[0]);

	dFdZ.resize(nLayer);
	for (int i = 0; i < nLayer; ++i) {
		dFdZ[i].resize(dimA[i], tSample);
	}
	dFdZ[nLayer - 1].setOnes();

	W.resize(nLayer);
	b.resize(nLayer);
	W_copy.resize(nLayer);
	b_copy.resize(nLayer);

	for (int i = 1; i < nLayer; ++i) {
		W[i].resize(dimA[i], dimA[i - 1]);
		b[i].resize(dimA[i]);
		W_copy[i].resize(dimA[i], dimA[i - 1]);
		b_copy[i].resize(dimA[i]);
	}

	nWeight = 0;
	for (int i = 1; i < nLayer; ++i) {
		nWeight += dimA[i] * (dimA[i - 1] + 1);
	}

	Jac.resize(tSample, nWeight);
	JtJ.resize(nWeight, nWeight);
	JtErr.resize(nWeight);
	dWeight.resize(nWeight);
}

void Network::DataInput(istream & fin)
{
	MatrixXd rawX(dimA[0], nSample);
	RowVectorXd rawE(nSample);

	for (int iSample = 0; iSample < nSample; ++iSample) {
		for (int i = 0; i < dimA[0]; ++i) {
			fin >> rawX(i, iSample);
		}
		fin >> rawE(iSample);
	}

	maxX = rawX.rowwise().maxCoeff();
	avgX = rawX.rowwise().mean();
	minX = rawX.rowwise().minCoeff();

	maxE = rawE.maxCoeff();
	avgE = rawE.mean();
	minE = rawE.minCoeff();

	rawX = (rawX.colwise() - avgX).array().colwise() / (maxX - minX).array();
	rawE = (rawE.array() - avgE) / (maxE - minE);

	vector<int> rand_seq(nSample);
	for (int i = 0; i < nSample; ++i) {
		rand_seq[i] = i;
	}
	random_shuffle(rand_seq.begin(), rand_seq.end());

	for (int i = 0; i < tSample; ++i) {
		tA[0].col(i) = rawX.col(rand_seq[i]);
		tE(i) = rawE(rand_seq[i]);
	}

	for (int i = tSample; i < nSample; ++i) {
		vA[0].col(i - tSample) = rawX.col(rand_seq[i]);
		vE(i - tSample) = rawE(rand_seq[i]);
	}
}

void Network::ForwardProp(vector<Eigen::MatrixXd> & A)
{
	for (int i = 1; i < nLayer - 1; ++i) {
		A[i] = ((W[i] * A[i - 1]).colwise() + b[i]).array() /
			sqrt(((W[i] * A[i - 1]).colwise() + b[i]).array().square() + 1);
	}

	A[nLayer - 1] = (W[nLayer - 1] * A[nLayer - 2]).colwise() + b[nLayer - 1];
}

void Network::BackwardProp()
{
	for (int i = nLayer - 2; i > 0; --i) {
		dFdZ[i] = (W[i + 1].transpose() * dFdZ[i + 1]).array() * 
			(1 - tA[i].array().square()).array().pow(1.5);
	}

	int iCol = 0;
	int pre_dim, dim;
	Jac.setZero();

	for (int iLayer = 1; iLayer < nLayer - 1; ++iLayer) {
		pre_dim = dimA[iLayer - 1];
		dim = dimA[iLayer];
		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < pre_dim; ++j) {
				Jac.col(iCol + i * pre_dim + j).array() +=
					(dFdZ[iLayer].array().row(i) * tA[iLayer - 1].array().row(j)).transpose();
			}
		}
		iCol += pre_dim * dim;
		Jac.block(0, iCol, tSample, dim) += dFdZ[iLayer].transpose();
		iCol += dim;
	}

	dim = dimA[nLayer - 2];
	Jac.block(0, iCol, tSample, dim) += tA[nLayer - 2].transpose();
	iCol += dim;

	Jac.col(iCol) = VectorXd::Ones(tSample);

	JtJ = Jac.transpose().eval() * Jac;
	JtErr = Jac.transpose() * tErr.transpose();

}

void Network::InitWeight()
{
	for (int i = 1; i < nLayer; ++i) {
		W[i].setRandom();
		b[i].setRandom();
	}
}

void Network::UpdateWeight(const double & mu)
{
	dWeight = (JtJ + mu * MatrixXd::Identity(nWeight, nWeight)).llt().solve(JtErr);

	int iRow = 0;
	int nRow;	

	for (int iLayer = 1; iLayer < nLayer; ++iLayer) {
		nRow = dimA[iLayer] * dimA[iLayer - 1];
		W[iLayer] += Map<Matrix<double, Dynamic, Dynamic, RowMajor>>
			(dWeight.segment(iRow, nRow).data(), dimA[iLayer], dimA[iLayer - 1]);
		iRow += nRow;
		b[iLayer] += dWeight.segment(iRow, dimA[iLayer]);
		iRow += dimA[iLayer];
	}
}

bool Network::CalcRMSE(const bool & train)
{
	bool state;	// if bad, stat = true

	if (train) {
		tErr = tE - tA[nLayer - 1];
		tRMSE = sqrt(tErr.squaredNorm() / tSample) * 1000;

		if (tRMSE > tRMSE_last) {
			state = true;
		}
		else {
			state = false;
			tRMSE_last = tRMSE;
		}
	}
	else {
		vErr = vE - vA[nLayer - 1];
		vRMSE = sqrt(vErr.squaredNorm() / vSample) * 1000;

		if (vRMSE > vRMSE_last) {
			state = true;
		}
		else {
			state = false;
			vRMSE_last = vRMSE;
		}
	}

	return state;
}

void Network::BackupWeight()
{
	for (int iLayer = 1; iLayer < nLayer; ++iLayer) {
		W_copy[iLayer] = W[iLayer];
		b_copy[iLayer] = b[iLayer];
	}
}

void Network::RestoreWeight()
{
	for (int iLayer = 1; iLayer < nLayer; ++iLayer) {
		W[iLayer] = W_copy[iLayer];
		b[iLayer] = b_copy[iLayer];
	}
}

void Network::Train(ostream & fout)
{
	bool terminate = false;
	int inc_step = 0;

	fout << setw(8) << left << "Epoch";
	fout << setw(12) << left << "tRMSE";
	fout << setw(12) << left << "vRMSE";
	fout << setw(8) << left << "mu" << endl;

	double mu = mu_init;

	InitWeight();
	ForwardProp(tA);
	CalcRMSE(true);
	ForwardProp(vA);
	CalcRMSE(false);

	tRMSE_last = tRMSE;
	vRMSE_last = vRMSE;

	fout << setw(8) << left << 0;
	fout << setw(12) << left << tRMSE;
	fout << setw(12) << left << vRMSE;
	fout << setw(8) << mu << endl;

	for (int iEpoch = 1; iEpoch <= 5000; ++iEpoch) {
	
		BackupWeight();
		BackwardProp();
		UpdateWeight(mu);

		ForwardProp(tA);
		while (CalcRMSE(true)) {

			mu *= 2;
			if (mu > 1e10) {
				terminate = true;
				break;
			}

			RestoreWeight();
			UpdateWeight(mu);
			ForwardProp(tA);
		}

		ForwardProp(vA);		
		if (CalcRMSE(false)) {
			++inc_step;
		}
		else {
			inc_step = 0;
		}

		fout << setw(8) << left << iEpoch;
		fout << setw(12) << left << tRMSE;
		fout << setw(12) << left << vRMSE;
		fout << setw(8) << left << mu << endl;

		if (terminate || inc_step > early_stop) {
			break;
		}
		else {
			mu /= 2;
		}
	}
}

void Network::OutputNet(ostream & fout)
{
	int dimX = dimA[0];

	for (int i = 0; i < dimX; ++i) {
		fout << setw(25) << left << minX(i);
	}
	fout << endl;

	for (int i = 0; i < dimX; ++i) {
		fout << setw(25) << left << avgX(i);
	}
	fout << endl;

	for (int i = 0; i < dimX; ++i) {
		fout << setw(25) << left << maxX(i);
	}
	fout << endl;

	fout << minE << endl;
	fout << avgE << endl;
	fout << maxE << endl;

	for (int iLayer = 1; iLayer < nLayer; ++iLayer) {
		Map<RowVectorXd> mapW(W[iLayer].data(), W[iLayer].size());
		Map<RowVectorXd> mapb(b[iLayer].data(), b[iLayer].size());

		fout << mapW << endl;
		fout << mapb << endl;
	}
}