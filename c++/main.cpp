// IsingMonteCarlo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "IsingModel.h"

int main()
{
	double t[5] = { 2,2.2,2.4,2.6,2.8 };
	for (double i : t) {
		IsingModel model = IsingModel(50, i);
		model.start(500000, 0, 0);
		
	}
}

