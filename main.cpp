#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <numeric>
#include <unordered_set>

#include "world.h"
// #include "kdtree.h"
#include "KdTree.h"
#include "timing.h"
#include "common.h"


int main(int argc, char **argv) {
	StartupOptions opts = parseOptions(argc, argv);
	// const int seed = argc > 1 ? std::stoi(argv[1]) : 0;
	// srand(seed);
	
	World<Point2D> world;
	std::vector<Point2D> queries = \
		world.generateQueries(opts.numQueries, opts.spaceSize, opts.queryFile);
	// world.generateDiagonal(opts.numPoints, opts.spaceSize);
	// world.savePointsToFile("benchmark-files/random-50000");
	world.loadPointsFromFile(opts.inputFile); 

	Timer build_timer;
	// build k-d tree
	// kdt::KDTree<Point2D> kdtree(world.points);
	
	// KdTree<Point2D> kdtree(world.points);
	KdTree<Point2D> *kdTree;
	
	if (opts.is_parallel) {
		kdTree = new PkdTree<Point2D>(world.points);
	} else {
		kdTree = new SkdTree<Point2D>(world.points);
	}
	// PkdTree<Point2D> kdtree(world.points);
	double build_time = build_timer.elapsed();

	double traverse_time = 0.;
	std::vector<std::vector<Point2D>> results(queries.size());

	Timer traverse_timer;
	if (opts.is_parallel) {
		#pragma omp parallel for schedule(dynamic, 128) //reduction(+:traverse_time)
		for (int i = 0; i < (int) queries.size(); ++i) {
			const auto &query = queries[i];
			std::vector<Point2D> neibors = kdTree->knn_search(query, opts.k);
			// traverse_time += traverse_timer.elapsed();
			results[i] = neibors;
			// std::unordered_map<Point2D, int> map;

			// for (const auto &p : world.points) {
			// 	map[p] = 0;
			// }

			// for (const auto &p : result) {
			// 	map[p] = 1;
			// }
			// map[query] = 2;
		
			// Output to file
			// world.saveKNNResultToFile("../logs/diagonal-50000.txt", map);
			// world.saveKNNResultToFile(opts.outputFile, map, true);
		}
	} else {
		for (int i = 0; i < (int)queries.size(); ++i) {
			const auto &query = queries[i];
			std::vector<Point2D> neibors = kdTree->knn_search(query, opts.k);
			// traverse_time += traverse_timer.elapsed();
			results[i] = neibors;
		}
	}
	traverse_time = traverse_timer.elapsed();
	
	for (int i = 0; i < (int) queries.size(); ++i) {
		std::sort(results[i].begin(), results[i].end()); // ! default: sort on first dim
		world.saveKNNResultToFile(opts.outputFile, queries[i], results[i]);
	}

	// delete kdTree;
	printf("build time: %.6fs\n", build_time);
	printf("traverse time: %.6fs\n", traverse_time);
	printf("is parallel: %d\n", opts.is_parallel);
	return 0;
}