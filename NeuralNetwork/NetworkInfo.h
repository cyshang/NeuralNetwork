#ifndef NETWORKINFO_H_
#define NETWORKINFO_H_

#include <vector>

struct NetworkInfo
{
	int nLayer;
	std::vector<int> dimA;
	int nSample;
	double ratio;
	double mu_init;
	int early_stop;
};

#endif // !NETWORKINFO_H_
