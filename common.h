#ifndef COMMON_H_
#define COMMON_H_

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct StartupOptions {
  int numIterations = -1;
  int numPoints = -1;
  float spaceSize = -1.f;
  bool loadBalance = false;
  bool is_parallel = false;
  std::string outputFile;
  std::string inputFile;
  std::string queryFile;
  int k = -1;
  int numQueries = -1;
};

inline StartupOptions parseOptions(int argc, char *argv[]) {
  StartupOptions rs;
  for (int i = 1; i < argc; i++) {
    if (i < argc - 1) {
      if (strcmp(argv[i], "-i") == 0)
        rs.numIterations = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-s") == 0)
        rs.spaceSize = (float)atof(argv[i + 1]);
      else if (strcmp(argv[i], "-in") == 0)
        rs.inputFile = argv[i + 1];
      else if (strcmp(argv[i], "-n") == 0)
        rs.numPoints = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-o") == 0)
        rs.outputFile = argv[i + 1];
      else if (strcmp(argv[i], "-q") == 0)
        rs.queryFile = argv[i + 1];
      else if (strcmp(argv[i], "-nq") == 0)
        rs.numQueries = atoi(argv[i + 1]);
      else if (strcmp(argv[i], "-k") == 0)
        rs.k = atoi(argv[i + 1]);
    }
    if (strcmp(argv[i], "-p") == 0)
      rs.is_parallel = true;
    if (strcmp(argv[i], "-lb") == 0) {
      rs.loadBalance = true;
    }
  }
  return rs;
}
#endif
