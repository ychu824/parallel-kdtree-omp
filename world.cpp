#include "world.h"
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <sys/stat.h>
#include <unistd.h>

#define MAGIC_NUMBER 10000
#define BIGLITTLE_RATIO (1.f / 3. * 2.)

/// @brief Random class from asst 3
class Random
{
private:
    unsigned int seed;

public:
    Random(int seed) : seed(seed) {}
    int Next() // random between 0 and RandMax (currently 0x7fff)
    {
        return (((seed = seed * 214013L + 2531011L) >> 16) & 0x7fff);
    }
    int Next(int min, int max) // inclusive min, exclusive max
    {
        unsigned int a = ((seed = seed * 214013L + 2531011L) & 0xFFFF0000);
        unsigned int b = ((seed = seed * 214013L + 2531011L) >> 16);
        unsigned int r = a + b;
        return min + r % (max - min);
    }
    float NextFloat() { return ((Next() << 15) + Next()) / ((float)(1 << 30)); }
    float NextFloat(float valMin, float valMax)
    {
        return valMin + (valMax - valMin) * NextFloat();
    }
    static int RandMax() { return 0x7fff; }
};

template <class PointT>
bool World<PointT>::loadPointsFromFile(std::string fileName) {
    std::ifstream inFile(fileName);
    assert((bool)inFile && "Cannot open input file");

    std::string line;
    points.resize(0);
    while (std::getline(inFile, line)) {
        PointT point;
        std::stringstream sstream(line);
        std::string str;
        for (int i = 0; i < PointT::DIM; ++i) {
            if (i == (PointT::DIM - 1)) {
                std::getline(sstream, str, '\n');
            } else {
                std::getline(sstream, str, ',');
            }
            point[i] = (float) atof(str.c_str());
        }
        points.push_back(point);
    }
    inFile.close();
    return true;
}

template <class PointT>
void World<PointT>::savePointsToFile(const std::string &fileName) {
    std::ofstream file(fileName);
    if (!file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return;
    }
    file << std::setprecision(9);
    for (const auto &p : points) {
        for (int i = 0; i < PointT::DIM; ++i) {
            if (i == (PointT::DIM - 1)) {
                file << p[i] << std::endl;
            } else {
                file << p[i] << ",";
            }
        }
    }
    file.close();
    if (!file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
}

template <class PointT>
void World<PointT>::saveKNNResultToFile(const std::string &fileName, const PointT &query,
    const std::vector<PointT> neibors) {
    std::ofstream output_file(fileName, std::ios::app);
    if (!output_file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return;
    }
    output_file << std::setprecision(9);
	for (int j = 0; j < PointT::DIM; ++j) {
		output_file << query[j] << ",";
	}
	output_file << "2" << std::endl;

	//Output all points
	for (int i = 0; i < (int) neibors.size(); ++i) {
		for (int j = 0; j < PointT::DIM; ++j) {
			output_file << neibors[i][j] << ",";
		}
		output_file << "1" << std::endl;
	}
	
	output_file.close();
    if (!output_file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
}

template <class PointT>
void World<PointT>::saveKNNResultToFile(const std::string &fileName, 
    const std::unordered_map<PointT, int> &map, bool save_neighbors_only) {
    std::ofstream output_file(fileName, std::ios::app);
    if (!output_file) {
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
        return;
    }

    output_file << std::setprecision(9);
    // Output all points
    for (const auto &[k, v] : map) {
        if (save_neighbors_only && v == 0) continue;
        for (int j = 0; j < PointT::DIM; ++j) {
            output_file << k[j] << ",";
        }

        output_file << v << std::endl;
    }

    output_file.close();
    if (!output_file)
        std::cout << "error writing file \"" << fileName << "\"" << std::endl;
}

template <class PointT>
void World<PointT>::generateRandom(int numPoints, float spaceSize) {
    Random random(MAGIC_NUMBER);
    points.resize(numPoints);

    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < PointT::DIM; ++j) {
            points[i][j] = random.NextFloat(0., spaceSize);
        }
    }
}

template <class PointT>
void World<PointT>::generateBigLittle(int numPoints, float spaceSize) {
    Random random(MAGIC_NUMBER);
    points.resize(numPoints);
    int numPartitions = (1 << PointT::DIM);
    int largeNumPoints = numPoints * (BIGLITTLE_RATIO);
    int smallNumPoints = (numPoints - largeNumPoints + numPartitions) / (numPartitions - 1);
    float halfSpaceSize = spaceSize / 2;

    // lower left = large cluster
    int i = 0;
    for (; i < largeNumPoints; i++) {
        for (int j = 0; j < PointT::DIM; ++j) {
            points[i][j] = random.NextFloat(0., halfSpaceSize);
        }
    }

    for (int k = 1; k < numPartitions; ++k) {
        int bound = std::min(largeNumPoints + (smallNumPoints * k), numPoints);
        for (; i < bound; i++) {
            for (int j = 0; j < PointT::DIM; ++j) {
                if (((k >> j) & 0x1) == 0x1) {
                    points[i][j] = random.NextFloat(halfSpaceSize, spaceSize);
                } else {
                    points[i][j] = random.NextFloat(0., halfSpaceSize);
                }
            }
        }
    }

    // // upper left = small cluster
    // for (; i < largeNumPoints + smallNumPoints; i++) {
    //     points[i][0] = random.NextFloat(0., halfSpaceSize);
    //     points[i][1] = random.NextFloat(halfSpaceSize, spaceSize);
    // }

    // // lower left = small cluster
    // for (; i < largeNumPoints + smallNumPoints * 2; i++) {
    //     points[i][0] = random.NextFloat(0., halfSpaceSize);
    //     points[i][1] = random.NextFloat(0., halfSpaceSize);
    // }

    // // lower right = small cluster
    // for (; i < numPoints; i++) {
    //     points[i][0] = random.NextFloat(halfSpaceSize, spaceSize);
    //     points[i][1] = random.NextFloat(0., halfSpaceSize);
    // }
}

template <class PointT>
void World<PointT>::generateDiagonal(int numPoints, float spaceSize) {
    Random random(MAGIC_NUMBER);
    points.resize(numPoints);

    float diagonalCenter = 0;
    float range = spaceSize / 10;
    int step = 5;

    for (int i = 0; i < numPoints; i++) {
        // compute left and right bounds for point to be generated in
        float leftBound = diagonalCenter - range;
        float rightBound = diagonalCenter + range;

        // clamp values
        if (leftBound < 0.)
            leftBound = 0.;
        if (rightBound > spaceSize)
            rightBound = spaceSize;

        for (int j = 0; j < PointT::DIM; ++j) {
            points[i][j] = random.NextFloat(leftBound, rightBound);
        }

        diagonalCenter += step;
        if (diagonalCenter > spaceSize) {
            diagonalCenter = 0.f;
        }
    }
}

template <class PointT>
std::vector<PointT> World<PointT>::generateQueries(int numQueries,
    float spaceSize, const std::string &fileName) { 
    std::vector<PointT> queries;
    bool is_exist = access(fileName.c_str(), F_OK) != -1;

    if (!is_exist) {
        Random random(MAGIC_NUMBER);
        points.resize(numQueries);

        for (int i = 0; i < numQueries; ++i) {
            for (int j = 0; j < PointT::DIM; ++j) {
                points[i][j] = random.NextFloat(0., spaceSize);
            }
        }

        savePointsToFile(fileName);
    } else {
        loadPointsFromFile(fileName);
    }

    queries.swap(points);
    return queries;
}

template class World<Point2D>;
template class World<Point3D>;