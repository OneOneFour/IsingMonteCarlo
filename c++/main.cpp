// IsingMonteCarlo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "IsingModel.h"
#include "C2PyPlot.h"
#define ARR_LEN 100

// YES I KNOW THIS IS RAM INEFFICENT BLAH BLAH BLAH 
std::vector<double> linspace(const double start, const double end, const int n_samples= 2) {
	std::vector<double> a(n_samples);
	double val;
	double step = (end - start) / n_samples;
	for (double i = 0,val=start; (int)i < n_samples;i++,val+=step) {
		a[(int)i] = val;
	}
	return a;
}

int main()
{
	std::vector<double> t = linspace(1.0, 5.0, ARR_LEN);
	std::vector<double> e(ARR_LEN);
	std::vector<double> m(ARR_LEN);
	std::vector<double> chi(ARR_LEN);
	std::vector<double> cv(ARR_LEN);
	Plotter p;
#pragma omp parallel for 
	for (int i = 0; i < ARR_LEN; i++) {
		IsingModel model(50, t[i]);
		model.start(100000000, 0, 0);
#pragma omp critical
		{
			std::cout << "Temperature: " << t[i] << std::endl;
			
			e[i] = model.get_mean_energy();
			std::cout << "Average E:" << e[i] << std::endl;
			m[i] = model.get_abs_mean_magnitization();
			std::cout << "Average m:" << m[i] << std::endl;

			chi[i] = model.get_m_variance();
			cv[i] = model.get_e_variance();
		}
	}
	//p.plot('t', 'e', &t, &e, "Energy");
	p.plot('t', 'm', &t,&m, "Magnetization");
	//p.plot('t', 'x', &t, &chi, "Succetibility");
	//p.plot('t', 'c', &t, &cv, "Heat Capacity");
	p.show();
	return 0;
}

