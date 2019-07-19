// IsingMonteCarlo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "IsingModel.h"


int main()
{
	double t[4] = {
		1.5,
		2.0,
		2.5,
		3.0
	};
	for (double i : t) {
		IsingModel model(50, i);
		std::cout << "Temperature: " <<i << std::endl;
		model.start(20000000, 0, 0);
		std::cout << "Average E:" << model.get_mean_energy() << std::endl;
		std::cout << "Average m:" << model.get_abs_mean_magnitization() << std::endl;



	}
	return 0;
}

