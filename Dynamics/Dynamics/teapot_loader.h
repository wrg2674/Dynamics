#ifndef TEAPOT_H
#define TEAPOT_H
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
using namespace std;

class Teapot
{
public:
	unsigned int nVertFloats;
	unsigned int nVertNum;
	bool err = 0;
	Teapot(const char* filename, std::vector<float> &data, unsigned int vertAttrib)
	{
		nVertFloats = vertAttrib;
		err = loadVertexData(string(filename), data);
		nVertNum = int(size(data) / nVertFloats);
	};
	bool loadVertexData(std::string filename, std::vector<float> &data) {
		// read vertex data from vbo file in plain text format
		ifstream input(filename.c_str());
		if (!input) { // cast istream to bool to see if something went wrong
			cerr << "Can not find vertex data file " << filename << endl;
			return false;
		}

		int numFloats;
		double vertData;
		if (input >> numFloats) {
			if (numFloats > 0) {
				data.resize(numFloats);
				int i = 0;
				while (input >> vertData && i < numFloats) {
					// store it in the vector
					data[i] = float(vertData);
					i++;
				}
				if (i != numFloats || numFloats % nVertFloats) return false;
			}
		}
		else {
			return false;
		}
		return true;
	}

};
#endif

