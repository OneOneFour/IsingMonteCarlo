#pragma once
#include <random> 
#include <math.h>
class IsingModel {
public:
	IsingModel(int size, double t);
	~IsingModel();
	void start(int max_iterations,int record_every,int delay);
	double calc_energy();
	double calc_magnetization();

	double* get_energy(); 
	double* get_magnetization();

	double get_mean_energy();
	double get_abs_mean_magnitization();
private:
	int metropolis_step(int i);
	//int wolff_step();
	int get_site(int x, int y);
	void flip_site(int x, int y);
	int size;
	int arrLen; 
	bool* lattice;
	double t;
	double* energy;
	double* magnetization;
	std::random_device rd;
	std::mt19937 rng;
	std::uniform_int_distribution<int> uni;
	std::uniform_real_distribution<double> r;
};

