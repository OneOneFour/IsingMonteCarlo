#pragma once
#ifndef C2_PY_PLOT_H
#define C2_PY_PLOT_H

#include <algorithm>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#define RED "r-"
#define BLUE "b-"
#define GREEN "g-"
class Plotter {
public:
	Plotter() {}
	~Plotter() {}
	void plot(char xChar, char yChar, const std::vector<double>* x = nullptr, const std::vector<double>* y = nullptr, std::string legend = "") {
		std::string str;
		str += xChar;
		str += yChar;
		if (std::find(__pointMap.begin(), __pointMap.end(), str) == __pointMap.end()) {
			if (x != nullptr) {
				__dataMap[xChar] = x;
			}
			if (y != nullptr) {
				__dataMap[yChar] = y;
			}
			__pointMap.push_back(str);
			if (legend != "") {
				__legend[str] = legend;
			}
		}
		else {
			throw std::logic_error("Combination already exists - key pair or use alternative var names");
		}
	}
#ifdef TASK_3_H
	void plot(const XYData& data, char xChar = 'x', char yChar = 'y') {
		plot(xChar, yChar, &data.getX(), &data.getY());
	}
#endif
	void imshow(const std::vector<double>& data, int width, int height, std::string label = "", double min_xExtent = 0.0, double min_yExtent = 0.0, double max_xExtent = 1.0, double max_yExtent = 1.0, std::string path = "output.py") {
		if (width * height != data.size()) {
			throw std::logic_error("Incompatible dimensions. Expected " + std::to_string(width * height) + ". Got " + std::to_string(data.size()));
		}
		std::fstream stream;
		stream.open(path, std::ios::out);
		stream << "import numpy as np\n" << "import matplotlib.pyplot as plt\n";
		stream << "data = np.array([";
		for (int i = 0, y = 0; y < height; y++) {
			stream << "[";
			for (int x = 0; x < width; x++, i++) {
				stream << data[i] << ",";
			}
			stream << "],\n";
		}
		stream << "])\n";
		stream << "fig,ax = plt.subplots()\n";
		stream << "im  = ax.imshow(data,cmap=plt.get_cmap(\"hot\"),interpolation='nearest',extent=(" << min_xExtent << "," << max_xExtent << "," << min_yExtent << "," << max_yExtent << "))\n";
		stream << "fig.colorbar(im)\n";
		stream << "ax.set_title(\"" << label << "\")\n";
		stream << "ax.set_xlabel(\"width(mm)\")\n";
		stream << "ax.set_ylabel(\"height(mm)\")\n";
		stream << "plt.show()\n";
		stream.close();
		system("python output.py");
	}
	void hist(char dChar, std::vector<double>* data, int bins) {
		this->bins = bins;
		__dataMap[dChar] = data;
		__histMap.push_back(dChar);
	}
	void save_file(std::string path = "output.py") {
		std::fstream stream;
		stream.open(path, std::ios::out);
		stream << "import numpy as np\n" << "import matplotlib.pyplot as plt\n";
		for (auto iter = __dataMap.begin(); iter != __dataMap.end(); ++iter) {
			stream << iter->first << "=[";
			for (unsigned int i = 0; i < iter->second->size(); i++) {
				stream << (*iter->second)[i] << ",";
			}
			stream << "]" << std::endl;
		}
		if (__pointMap.size() > 0) {
			for (auto pair : __pointMap) {
				stream << "plt.plot(" << pair[0] << "," << pair[1];
				if (__legend.find(pair) != __legend.end()) {
					stream << ",label=\"" << __legend[pair].c_str() << "\"";
				}
				stream << ")\n";
			}
			stream << "plt.legend()\nplt.show()" << std::endl;
		}
		if (__histMap.size() > 0) {
			for (unsigned int i = 0; i < __histMap.size(); i++) {

				stream << "plt.hist(" << __histMap[i] << ",bins=" << bins << ")\n";
			}
			stream << "plt.show()\n";
		}
		stream.close();
	}
	void show() {
		save_file();
		system("python output.py");
	}
	void clear() {
		__dataMap.clear();
		__pointMap.clear();
		__histMap.clear();
		__legend.clear();
	}
private:
	int bins;
	std::map<char, const std::vector<double>*> __dataMap;
	std::vector<std::string> __pointMap;
	std::vector<char> __histMap;
	std::map <std::string, std::string> __legend;
};
#endif#pragma once
