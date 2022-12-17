#ifndef WORLD_H
#define WORLD_H
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>

class Point2D : public std::array<float, 2> {
public:
    static const int DIM = 2;

    static int dim() {
        return DIM;
    }

    Point2D() {
        (*this)[0] = 0;
        (*this)[1] = 0;
    }

    Point2D(float x, float y) {
        (*this)[0] = x;
        (*this)[1] = y;
    }
};

template <>
struct std::hash<Point2D> {
    std::size_t operator()(Point2D const &p) const noexcept {
        uint64_t ret;
        ret = *(uint64_t *)p.data();
        return ret;
    }
};

class Point3D : public std::vector<float> {
public:
    static const int DIM = 3;

    static int dim() {
        return DIM;
    }

    Point3D() {
        this->resize(DIM);
    }
    Point3D(float x, float y, float z) {
        this->resize(DIM);
        (*this)[0] = x;
        (*this)[1] = y;
        (*this)[2] = z;
    }
};

template <>
struct std::hash<Point3D> {
    std::size_t operator()(Point3D const &p) const noexcept {
        size_t h1 = p[0];
        size_t h2 = p[1];
        size_t h3 = p[2];
        return h3 ^ ((h2 ^ (h1 << 1)) << 1);
    }
};

template <class PointT>
class World {
public:
    std::vector<PointT> points;
    bool loadPointsFromFile(std::string fileName);
    void savePointsToFile(const std::string &fileName);
    static void saveKNNResultToFile(const std::string &fileName, const PointT &query, 
                             const std::vector<PointT> neibors);
    void saveKNNResultToFile(const std::string &fileName, 
                             const std::unordered_map<PointT, int> &map, bool save_neighbors_only);
    void generateRandom(int numPoints, float spaceSize);
    void generateBigLittle(int numPoints, float spaceSize);
    void generateDiagonal(int numPoints, float spaceSize);
    std::vector<PointT> generateQueries(int numQueries, float spaceSize, 
        const std::string &fileName);
};
#endif