//
// Created by Yucheng on 2022/12/10.
//

#ifndef KD_TREE_KDTREE_H
#define KD_TREE_KDTREE_H
#include <vector>
#include <cmath>
#include <mutex>
#include <algorithm>
#include "omp.h"

template<class T>
static double distance(const T &p1, const T &p2) {
    double dist = 0.;
    for (int i = 0; i < T::dim(); ++i) {
        dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }

    return std::sqrt(dist);
}

template<class T>
struct KdTreeNode {
    T point;
    KdTreeNode *left {}, *right {};
    int axis {};

    KdTreeNode() = default;
    KdTreeNode(const T &p, KdTreeNode *l, KdTreeNode *r, int a): point(p), left(l), right(r), axis(a) {}
};

template<class T>
class KdTree {
public:
    virtual std::vector<T> knn_search(const T &query, int k) = 0;
    virtual ~KdTree() = default;
};

template<class T>
class SkdTree: public KdTree<T> {
public:
    SkdTree() = default;
    explicit SkdTree(const std::vector<T> &points);
    std::vector<T> knn_search(const T &query, int k);
    ~SkdTree() = default;
private:
    void knn_search_helper(const KdTreeNode<T> *node, const T& query,
                                     std::vector<T> &heap, int k);
    KdTreeNode<T> *build_helper(std::vector<T> points, int depth);
    KdTreeNode<T> *root {};
};

template <class T>
class PkdTree: public KdTree<T> {
public:
    PkdTree() = default;
    explicit PkdTree(const std::vector<T> &points);
    std::vector<T> knn_search(const T &query, int k);
    ~PkdTree() = default;
private:
    void knn_search_helper_v1(const KdTreeNode<T> *node, const T &query,
                           std::vector<T> &heap, int k, int threads_left);
    KdTreeNode<T> *build_helper(std::vector<T> points, int depth, int threads_left);
    KdTreeNode<T> *root {};
};

#endif //KD_TREE_KDTREE_H

