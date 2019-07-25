#pragma once
#include <random> 
#include <string>
#include <math.h>
#include <vector>

std::string get_json_str(bool* arr, int size, bool supercrit);

class IsingModel {
public:
	IsingModel(int size, double t,int iterations);
	~IsingModel();
	void start(int record_every,int delay);
	double calc_energy();
	double calc_magnetization();

	double get_mean_energy();
	double get_abs_mean_magnitization();

	bool get_supercritical();

	double get_e_variance();
	double get_m_variance();

	std::vector<bool*> record_states;
private:
	int metropolis_step(int i);
	//int wolff_step();
	int get_site(int x, int y);
	void flip_site(int x, int y);
	int size;
	bool* lattice;
	double t;
	double e;
	double m;
	double sum_m;
	double sum_e;
	double sum_esq;
	double sum_msq;
	double abs_m;
	int iterations;
	std::random_device rd;
	std::mt19937 rng;
	std::uniform_int_distribution<int> uni;
	std::uniform_real_distribution<double> r;
};

