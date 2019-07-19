#include "IsingModel.h"


IsingModel::IsingModel(int size, double t) :rng{ rd() }, uni(0, size), r(0, 1),arrLen(0),energy(nullptr),magnetization(nullptr) {
	this->size = size;
	this->t = t;
	this->lattice = new bool[this->size * this->size];

	for (int i = 0; i < this->size*this->size; i++) {
		this->lattice[i] = false;
	}


}
IsingModel::~IsingModel() {
	delete[] this->lattice;
	delete[] this->energy;
	delete[] this->magnetization;
}
int IsingModel::get_site(int x, int y) {
	// Implicitly Perodic conditions - if x/y out of range, wrap to be in the lattice somewhere (infinite lattice appx) 
	int x_p = x % this->size;
	int y_p = y % this->size;
	int result = this->lattice[x_p + y_p * this->size];
	return result * 2 - 1;
}

void IsingModel::flip_site(int x, int y) {
	int x_p = x % this->size;
	int y_p = y % this->size;
	this->lattice[x_p + y_p * this->size] = !this->lattice[x_p + y_p * this->size];
}

void IsingModel::start(int max_iterations, int sample_every, int delay) {
	if (energy != nullptr) delete[] this->energy;
	if (magnetization != nullptr) delete[] this->magnetization;
	this->energy = new double[max_iterations - delay];
	this->magnetization = new double[max_iterations - delay];
	this->arrLen = max_iterations - delay;
	this->energy[0] = this->calc_energy();
	this->magnetization[0] = this->calc_magnetization();
	int i = 1;
	while (i < max_iterations) {
		int iter_count = this->metropolis_step(i);
		i += iter_count;
	}
}

double IsingModel::calc_energy() {
	double E = 0.0;
	for (int y = 0; y < this->size; y++) {
		for (int x = 0; x < this->size; x++) {
			E += -1.0 * this->get_site(x, y) * (
				this->get_site(x + 1, y) +
				this->get_site(x - 1, y) +
				this->get_site(x, y + 1) +
				this->get_site(x, y - 1));
		}
	}
	return E;
}

double IsingModel::calc_magnetization() {
	double m = 0;
	for (int i = 0; i < this->size * this->size; i++) {
		m += (int)this->lattice[i] * 2 - 1;
	}
	return m;
}

double* IsingModel::get_energy()
{
	return this->energy;
}

double* IsingModel::get_magnetization()
{
	return this->magnetization;
}

double IsingModel::get_mean_energy()
{
	double e_bar = 0;
	for (int i = 0; i < this->arrLen; i++) {
		e_bar += this->energy[i];
	}
	return (double)e_bar / ((double)(this->size*this->size) * this->arrLen);
}

double IsingModel::get_abs_mean_magnitization()
{
	double m_bar = 0.0;
	for (int i = 0; i < this->arrLen; i++) {
		m_bar += this->magnetization[i];
	}
	double denom = (double)(this->size * this->size) * this->arrLen;
	return abs(m_bar) / denom;

}

int IsingModel::metropolis_step(int i) {
	int randX = this->uni(this->rng);
	int randY = this->uni(this->rng);
	int deltaE = 2 * this->get_site(randX, randY) * (
		this->get_site(randX + 1, randY) +
		this->get_site(randX - 1, randY) +
		this->get_site(randX, randY + 1) +
		this->get_site(randX, randY - 1)
		);
	double p = exp(-(double)deltaE / this->t);
	double r = this->r(this->rng);
	if (deltaE <= 0 || r <= p) {
		this->flip_site(randX, randY);
		this->energy[i] = this->energy[i - 1] + (double)deltaE;
		double dm = (2.0 * (double)this->get_site(randX, randY));
		this->magnetization[i] = this->magnetization[i - 1] + dm;
	}
	else {
		this->energy[i] = this->energy[i - 1];
		this->magnetization[i] = this->magnetization[i - 1];
	}
	return 1;
}

