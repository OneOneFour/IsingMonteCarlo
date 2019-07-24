#include "IsingModel.h"


IsingModel::IsingModel(int size, double t) :rng{ rd() }, uni(0, size-1), r(0, 1), e(0), m(0), last_e(0), last_m(0), esq(0), msq(0) {
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

void IsingModel::start(int max_iterations, int sample_every, int delay) {
	this->last_e = this->calc_energy();
	this->last_m = this->calc_magnetization();

	// Potential future optimization 
	//this->record_states.resize((max_iterations - delay) / sample_every);

	int i = 0;
	while (i < max_iterations) {
		int iter_count = this->metropolis_step(i - delay);
		i += iter_count;
		if (i >= delay && i % sample_every == 0) {
			bool* state = new bool[this->size * this->size];
			memcpy(state, this->lattice, this->size * this->size * sizeof(bool));
			this->record_states.push_back(state);
		}
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
	str += "]}";
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
	return this->e;
}

double IsingModel::get_abs_mean_magnitization()
{
	return abs(this->m) / ((double)this->size * (double)this->size);
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

		double dm = (2.0 * (double)this->get_site(randX, randY));
	
		this->last_e += deltaE;
		this->last_m += dm;
		
	}
	if (i == 0) {
		this->e = this->last_e;
		this->m = this->last_m;
		this->esq = this->last_e * this->last_e;
		this->msq = this->last_m * this->last_m;
	}
	else if (i > 0) {
		this->e += (this->last_e - this->e) / ((double)i);
		this->m += (this->last_m - this->m) / ((double)i);

		this->esq += (this->last_e * this->last_e - this->esq) / (double)i;
		this->msq += (this->last_m * this->last_m - this->msq) / (double)i;
	}
	return 1;
}


double IsingModel::get_e_variance() {
	return this->esq - (this->e * this->e);
}

double IsingModel::get_m_variance() {
	return this->msq - this->m * this->m;
}