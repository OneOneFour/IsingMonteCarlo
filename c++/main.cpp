// IsingMonteCarlo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "IsingModel.h"


int main()
{
	double t[5] = { 2,2.2,2.4,2.6,2.8 };
	for (double i : t) {
		IsingModel model(50, i);
		std::cout << "Temperature: " <<i << std::endl;
		model.start(2000000, 0, 0);
		std::cout << "Average E:" << model.get_mean_energy() << std::endl;
		std::cout << "Average m:" << model.get_abs_mean_magnitization() << std::endl;
	}
}

