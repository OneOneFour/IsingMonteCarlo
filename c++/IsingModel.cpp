#include "IsingModel.h"


IsingModel::IsingModel(int size, double t,int iterations) :rng{ rd() }, uni(0.0, size-1), r(0.0, 1.0), e(0),
															m(0), sum_esq(0), sum_msq(0),iterations(iterations),
															sum_m(0),sum_e(0),abs_m(0)
{
	this->size = size;
	this->t = t;
	this->lattice = new bool[this->size * this->size];

	for (int i = 0; i < this->size * this->size; i++) {
		this->lattice[i] = false;
	}


}
IsingModel::~IsingModel() {
	delete[] this->lattice;
	for (int i = 0; i < this->record_states.size(); i++) {
		delete[] this->record_states[i];
	}
}
int IsingModel::get_site(int x, int y) {
	// Implicitly Perodic conditions - if x/y out of range, wrap to be in the lattice somewhere (infinite lattice appx) 
	int x_p = (this->size + x % this->size)% this->size;
	int y_p = (this->size +y % this->size) %this->size;
	int result = this->lattice[x_p + y_p * this->size];
	return result * 2 - 1;
}

void IsingModel::flip_site(int x, int y) {
	int x_p = (this->size + x % this->size) % this->size;
	int y_p = (this->size + y % this->size) % this->size;
	this->lattice[x_p + y_p * this->size] = !this->lattice[x_p + y_p * this->size];
}

void IsingModel::start(int sample_every, int delay) {

	// Potential future optimization 
	//this->record_states.resize((max_iterations - delay) / sample_every);
	this->m = this->calc_magnetization();
	this->e = this->calc_energy();
	int i = 0;
	while (i < this->iterations) {
		this->metropolis_step(++i);
		if (i >= delay && i % sample_every == 0) {
			bool* state = new bool[this->size * this->size];
			memcpy(state, this->lattice, this->size * this->size * sizeof(bool));
			this->record_states.push_back(state);
		}
		this->sum_e += this->e;
		this->sum_m += this->m;
		this->sum_esq += this->e * this->e;
		this->sum_msq += this->m * this->m;
		this->abs_m += abs(this->m);
	}
}

std::string get_json_str(bool* arr, int size, bool supercrit) {
	std::string str = "{\"label\":";
	str += std::to_string((int)supercrit * 2 - 1);
	str += ",\"image\":[";
	for (int y = 0, i = 0; y < size; y++) {
		str += "[";
		for (int x = 0; x < size; x++, i++) {
			str += std::to_string(((int)arr[i]) * 2 - 1);
			if (x != size - 1) str += ",";
		}
		str += "],";

	}
	str += "],temp:";
	str += std::to_string(this->t);
	str+="}";
	return str;
}

bool IsingModel::get_supercritical() {
	return this->t > 2 / log(1 + sqrt(2));
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

double IsingModel::get_mean_energy()
{
	return this->sum_e/this->iterations;
}

double IsingModel::get_abs_mean_magnitization()
{
	return abs_m/ ((double)this->size * (double)this->size * (double)this->iterations);
}

int IsingModel::metropolis_step(int i ) {
	for (int j = 0; j < this->size * this->size; j++) {
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

			double dm = (2.0 * (double)this->get_site(randX, randY));

			this->e += deltaE;
			this->m += dm;

		}
	}
	return 1;
}


double IsingModel::get_e_variance() {
	return this->sum_esq /this->iterations - this->get_mean_energy() * this->get_mean_energy();
}

double IsingModel::get_m_variance() {
	return this->sum_msq / this->iterations - (sum_m / this->iterations) * (sum_m / this->iterations);
}