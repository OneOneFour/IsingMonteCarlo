// IsingMonteCarlo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <stack>
#include <nlohmann/json.hpp>
#include "IsingModel.h"
#include "C2PyPlot.h"

#define LATTICE_SIZE 50

using json = nlohmann::json;

constexpr auto ARR_LEN = 100;
const double T_CRIT = 2.0 / log(1.0 + sqrt(2.0));
const double RANGE = 5.0 / (4.0 * LATTICE_SIZE);

// YES I KNOW THIS IS RAM INEFFICENT BLAH BLAH BLAH 
std::vector<double> linspace(const double start, const double end, const int n_samples = 2) {
	std::vector<double> a(n_samples);
	double val;
	double step = (end - start) / (n_samples - 1);
	for (double i = 0, val = start; (int)i < n_samples; i++, val += step) {
		a[(int)i] = val;
	}
	return a;
}


void getData(const double start_temp, const double end_temp, const int N_steps, std::string path, int iterations = 1000000,
	int record_every = 200, int delay = 1000, int fileSize = 4000) {
	std::vector<double> t = linspace(start_temp, end_temp, N_steps);
	std::vector<double> e(N_steps);
	std::vector<double> m(N_steps);
	std::vector<double> chi(N_steps);
	std::vector<double> cv(N_steps);
	Plotter p;
	//main json file
	json core_json = json::array();
	int batch = 0;
#pragma omp parallel for
	for (int i = 0; i < N_steps; i++) {
		IsingModel model(LATTICE_SIZE, t[i], iterations);

		model.start(record_every, delay);
#pragma omp critical
		{

			std::cout << "Temperature: " << t[i] << std::endl;

			e[i] = model.get_mean_energy();
			std::cout << "Average E:" << e[i] << std::endl;
			m[i] = model.get_abs_mean_magnitization();
			std::cout << "Average m:" << m[i] << std::endl;

			chi[i] = model.get_m_variance() / t[i];
			cv[i] = model.get_e_variance() / (t[i] * t[i]);
			bool supercrit = t[i] >= 2.0 / log(1.0 + sqrt(2.0));
			for (bool* arr : model.record_states) {
				std::vector<std::vector<int>> shapedArr(50);
				for (int j = 0, l = 0; j < 50; j++) {
					shapedArr[j].resize(50);
					for (int k = 0; k < 50; k++, l++) {
						shapedArr[j][k] = arr[l] * 2- 1;
					}
				}
				json item;
				item["label"] = supercrit;
				item["temp"] = t[i];
				item["image"] = shapedArr;
				core_json.push_back(item);
				if (core_json.size() >= fileSize) {
					std::string tmppath = path;
					tmppath.insert(0, "batch_" + std::to_string(batch++) + "_");
					std::ofstream file(tmppath);
					file << core_json;
					core_json = json::array();
					file.close();
					std::cout << "File:" << tmppath << " successfully written";
				}
			}
		}


	}
	//p.plot('t', 'e', &t, &e, "Energy");
	//p.show();

	//p.plot('t', 'm', &t, &m, "Magnetization");
	//p.show();
	//p.plot('t', 'x', &t, &chi, "Succetibility");
	//p.show();
	//p.plot('t', 'c', &t, &cv, "Heat Capacity");
	//p.show();

	// file I/O timew
	std::string tmppath = path;
	tmppath.insert(0, "batch_" + std::to_string(batch++) + "_");
	std::ofstream file(tmppath);
    file.close();
	std::cout << "File:" << tmppath << " successfully written";


	// Write metadatafile
	json metaJson;
	metaJson["startTemp"] = start_temp;
	metaJson["startTempRangeSteps"] = (T_CRIT - start_temp)/RANGE
	metaJson["endTemp"] = end_temp;
	metaJson["endTempRangeSteps"] = (end_temp-T_CRIT)/RANGE
	metaJson["tempSteps"] = N_steps;
	metaJson["files"] = json::array();
    for(int i =0; i < batch; i++){
        metaJson["files"].push_back("batch_"+std::to_string(i)+path);
    }
    metaJson["iterations"] = iterations;
    metaJson["record_every"] = record_every;
    metaJson["delay"] = delay;
    metaJson["lattice_size"] = LATTICE_SIZE;
    metaJson["energy"] = e;
    metaJson["magnetization"] = m;
    metaJson["susceptibility"] = chi;
    metaJson["heat_capacity"] = cv;
	std::string metapath = "meta_"+ path;
	std::ofstream metafile(metapath);
    metafile << metaJson;
    file.close();
    std::cout << "File:" << tmppath << " successfully written";

}

int main()
{

	getData(T_CRIT - RANGE,T_CRIT + RANGE,2, "atcrit06082019.json",5000000,1000,1000);
	getData(T_CRIT - RANGE, T_CRIT +RANGE, 4, "probingcritical.json", 2500000,1000,1500);
	return 0;
}

