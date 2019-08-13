// IsingMonteCarlo.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <stack>
#include <filesystem>
#include <nlohmann/json.hpp>
#if defined(_WIN64)
#include <Windows.h>
#endif
#include "IsingModel.h"
#include "C2PyPlot.h"

using json = nlohmann::json;

constexpr auto ARR_LEN = 100;
const double T_CRIT = 2.0 / log(1.0 + sqrt(2.0));

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


void getData(const double start_temp, const double end_temp,int lattice_size, const int N_steps, std::string path, int iterations = 1000000,
	int record_every = 200, int delay = 1000, int fileSize = 4000) {
	path += ".json";
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
		IsingModel model(lattice_size, t[i], iterations);

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
					std::cout << "File:" << tmppath << " successfully written"<<std::endl;
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
	file << core_json; // lol should probably write the file!
    file.close();
	std::cout << "File:" << tmppath << " successfully written"<<std::endl;


	// Write metadatafile
	json metaJson;
	metaJson["startTemp"] = start_temp;
	metaJson["startTempRangeSteps"] = (T_CRIT - start_temp) / RANGE;
	metaJson["endTemp"] = end_temp;
	metaJson["endTempRangeSteps"] = (end_temp - T_CRIT) / RANGE;
	metaJson["tempSteps"] = N_steps;
	metaJson["files"] = json::array();
    for(int i =0; i < batch; i++){
        metaJson["files"].push_back("batch_"+std::to_string(i)+"_"+path);
    }
    metaJson["iterations"] = iterations;
    metaJson["record_every"] = record_every;
    metaJson["delay"] = delay;
    metaJson["lattice_size"] = lattice_size;
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
	// Begin work on some sort of user interface
	std::vector<json> batched_jobs;
	int bufferSize = GetCurrentDirectory(0, NULL);
	char* strBuffer = new char[bufferSize];
	GetCurrentDirectory(bufferSize, strBuffer);
	while (true) {
		json job;
		double startT, endT;
		int lattice_size;
		int N_samples, iterations, record_every, delay;
		std::string path, response;
		std::cout << "Enter lattice size:";
		std::cin >> lattice_size;
		do {
			response = "";
			std::cout << "Do you want to use absolute temperatures or those relative to critical point? (abs/rel/rel_range)  ";
			std::cin >> response;
		} while (response != "abs" && response != "rel" && response != "rel_range");
		if (response == "abs") {
			std::cout << "Enter starting temperature:  ";
			std::cin >> startT;

			std::cout << "Enter end temperature:  ";
			std::cin >> endT;
		}
		else if(response == "rel_range"){
			double RANGE = 5.0 / (4.0 * lattice_size);
			std::cout << "Enter starting temperature in units of \"range\" away from the critical temperature" <<
				std::endl << "Range is defined as 5/4*latticeSize" << std::endl << "The lattice size is " << LATTICE_SIZE << std::endl << "?  ";
			int tSteps; std::cin >> tSteps;
			startT = T_CRIT - RANGE * tSteps;

			std::cout << "Enter final temperature in units of \"range\" away from the critical temperature" <<
				std::endl << "Range is defined as 5/4*latticeSize" << std::endl << "The lattice size is " << LATTICE_SIZE << std::endl << "?  ";
			std::cin >> tSteps;
			endT = T_CRIT + RANGE * tSteps;
		}
		else {
			std::cout << "Enter temperature in units of Delta T away from the critical point (symmetric)";
			int deltaT; std::cin >> deltaT;
			startT = T_CRIT - deltaT;
			endT = T_CRIT + deltaT;
		}
		std::cout << "Enter the number of steps between the temperatures:  ";
		std::cin >> N_samples;

		std::cout << "Enter the number of Monte Carlo iterations to run each step for:  ";
		std::cin >> iterations;

		std::cout << "Enter the frequency with which to capture samples:  ";
		std::cin >> record_every;
		
		std::cout << "Enter the amount of Monte Carlo time to wait until capturing samples:  ";
		std::cin >> delay;
		std::cin.ignore();
		std::cout << "Enter the path to save files:	";
		std::getline(std::cin, path);
		std::filesystem::create_directory(path);

		std::cout << "Enter a name/description for this job (optional):  ";
		std::string desc;
		std::getline(std::cin, desc);
		std::cout << "Do you wish to add another batched job?(Y/N)  ";
		std::string resp; std::cin >> resp;


		job["startT"] = startT;
		jon["lattice_size"] = lattice_size;
		job["endT"] = endT;
		job["N_samples"] = N_samples;
		job["iterations"] = iterations;
		job["record_every"] = record_every;
		job["delay"] = delay;
		job["path"] = path;
		job["name"] = desc;
		batched_jobs.push_back(job);
		if (resp == "N") break;				
	}
	
	for (json& job :batched_jobs) {
		std::cout << "Executing job: " << job["name"] << std::endl;
		std::string ppd = job["path"];
		ppd.insert(0, "\\");
		std::string path = strBuffer + ppd;
		if (SetCurrentDirectory((path).c_str()) != 0 ) {
			getData(job["startT"], job["endT"],job["lattice_size"],job["N_samples"], job["path"], job["iterations"], job["record_every"], job["delay"]);
		}
		else {
			std::cout << "something bad happened when changing working directory" << std::endl;
		}
	}
	return 0;
}

