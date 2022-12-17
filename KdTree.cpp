//
// Created by Yucheng on 2022/12/10.
//
#include "world.h"
#include "KdTree.h"

template <class T>
SkdTree<T>::SkdTree(const std::vector<T> &points) {
    this->root = this->build_helper(points, 0);
}

template <class T>
KdTreeNode<T> *SkdTree<T>::build_helper(std::vector<T> points, int depth) {
    if (points.empty()) return nullptr;

    int k = points[0].size();
    int dim_to_divide = (depth % k);

    // FIXME: Use nth_element
    std::sort(points.begin(), points.end(), [&](const T &p1, const T &p2) {
        return p1[dim_to_divide] < p2[dim_to_divide];
    });

    int median = points.size() >> 1;
    auto start_left = points.cbegin();
    auto end_left = start_left + median;
    auto start_right = end_left + 1;
    auto end_right = points.cend();

    // FIXME: Do not copy to new memory
    auto *ret = new KdTreeNode<T>(
            points[median],
            build_helper(std::vector<T>(start_left, end_left), depth + 1),
            build_helper(std::vector<T>(start_right, end_right), depth + 1), dim_to_divide);
    return ret;
}

template <class T>
std::vector<T> SkdTree<T>::knn_search(const T &query, int k) {
    std::vector<T> ret;
    knn_search_helper(this->root, query, ret, k);
    return ret;
}

template <class T>
void SkdTree<T>::knn_search_helper(const KdTreeNode<T> *node, const T &query,
                                   std::vector<T> &heap, int k) {
    if (node == nullptr) return;

    heap.emplace_back(node->point);
    std::push_heap(heap.begin(), heap.end(), [&](const T &p1, const T &p2) {
        return distance(p1, query) < distance(p2, query);
    });

    if ((int) heap.size() > k) {
        std::pop_heap(heap.begin(), heap.end(), [&](const T &p1, const T &p2) {
            return distance(p1, query) < distance(p2, query);
        });
        heap.erase(heap.end() - 1);
    }

    int axis = node->axis;
    bool search_left = query[axis] < node->point[axis];
    // Search left subtree
    if (search_left) {
        knn_search_helper(node->left, query, heap, k);
    }
    // Search right subtree
    else {
        knn_search_helper(node->right, query, heap, k);
    }

    if ((int) heap.size() < k) {
        if (search_left) {
            knn_search_helper(node->right, query, heap, k);
        } else {
            knn_search_helper(node->left, query, heap, k);
        }
    } else {
        auto first = heap.front();
        double delta0 = std::fabs(node->point[axis] - query[axis]);
        double delta1 = std::fabs(first[axis] - query[axis]);
        if (delta0 < delta1) {
            if (search_left) {
                knn_search_helper(node->right, query, heap, k);
            } else {
                knn_search_helper(node->left, query, heap, k);
            }
        }
    }
}

template <class T>
PkdTree<T>::PkdTree(const std::vector<T> &points) {
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            int all_threads = omp_get_num_threads();
            
            this->root = this->build_helper(points, 0, all_threads);
        }
    }
}

template <class T>
KdTreeNode<T> *PkdTree<T>::build_helper(std::vector<T> points, int depth, int threads_left) {
    if (points.empty())
        return nullptr;

    int k = points[0].size();
    int dim_to_divide = (depth % k);

    // std::sort(points.begin(), points.end(), [&](const T &p1, const T &p2)
    //              { return p1[dim_to_divide] < p2[dim_to_divide]; });
#pragma region Osort
    if (threads_left <= 1) {
        std::sort(points.begin(), points.end(), [&](const T &p1, const T &p2)
                  { return p1[dim_to_divide] < p2[dim_to_divide]; });
    }
    else {
        int total_size = points.size();
        for (int i = 0; i < threads_left; ++i) {
            #pragma omp task shared(points, threads_left) 
            {
                int start = total_size * i / threads_left;
                int end = total_size * (i + 1) / threads_left;
                end = std::min(total_size, end);
                auto begin = std::begin(points);
                std::sort(begin + start, begin + end, [&](const T &p1, const T &p2)
                          { return p1[dim_to_divide] < p2[dim_to_divide]; });
            }
        }
        #pragma omp taskwait
        for (int step = 1; step < threads_left; step <<= 1) {
            for (int i = 0; i < (threads_left / 2 / step); ++i)
                #pragma omp task shared(step, points, threads_left)
                {
                    int start = total_size * (i * 2 * step) / threads_left;
                    int mid = total_size * (i * 2 * step + step) / threads_left;
                    int end = total_size * (i * 2 * step + 2 * step) / threads_left;
                    end = std::min(end, total_size);
                    auto begin = std::begin(points);
                    std::inplace_merge(begin + start, begin + mid, begin + end, [&](const T &p1, const T &p2)
                                    { return p1[dim_to_divide] < p2[dim_to_divide]; });
                }
            #pragma omp taskwait
        }
    }
#pragma endregion

    int median = points.size() >> 1;
    auto start_left = points.cbegin();
    auto end_left = start_left + median;
    auto start_right = end_left + 1;
    auto end_right = points.cend();

    KdTreeNode<T> *left_child, *right_child;

    if (threads_left <= 1) {
        left_child = this->build_helper(std::vector<T>(start_left, end_left), depth + 1, 1);
        right_child = this->build_helper(std::vector<T>(start_right, end_right), depth + 1, 1);
    }
    else {
        #pragma omp task shared(left_child)
            left_child = this->build_helper(std::vector<T>(start_left, end_left), depth + 1, threads_left / 2);
        #pragma omp task shared(right_child)
            right_child = this->build_helper(std::vector<T>(start_right, end_right), depth + 1, threads_left - threads_left / 2);
        #pragma omp taskwait
    }

    auto *ret = new KdTreeNode<T>(
        points[median],
        left_child, right_child, dim_to_divide);
    return ret;
}

template <class T>
std::vector<T> PkdTree<T>::knn_search(const T &query, int k) {
    std::vector<T> ret;
    this->knn_search_helper_v1(this->root, query, ret, k, 0);
    return ret;
}

template <class T>
void PkdTree<T>::knn_search_helper_v1(
    const KdTreeNode<T> *node, const T &query, std::vector<T> &heap, int k, int threads_left) {
    if (node == nullptr)
        return;

    heap.emplace_back(node->point);
    std::push_heap(heap.begin(), heap.end(), [&](const T &p1, const T &p2)
                   { return distance(p1, query) < distance(p2, query); });
    
    if ((int)heap.size() > k) {
        std::pop_heap(heap.begin(), heap.end(), [&](const T &p1, const T &p2)
                      { return distance(p1, query) < distance(p2, query); });
        heap.erase(heap.end() - 1);
    }

    int axis = node->axis;
    bool search_left = query[axis] < node->point[axis];
    // Search left subtree
    if (search_left) {
        knn_search_helper_v1(node->left, query, heap, k, threads_left);
    }
    // Search right subtree
    else {
        knn_search_helper_v1(node->right, query, heap, k, threads_left);
    }

    if ((int)heap.size() < k) {
        if (search_left) {
            knn_search_helper_v1(node->right, query, heap, k, threads_left);
        }
        else {
            knn_search_helper_v1(node->left, query, heap, k, threads_left);
        }
    }
    else {
        auto first = heap.front();
        double delta0 = std::fabs(node->point[axis] - query[axis]);
        double delta1 = std::fabs(first[axis] - query[axis]);
        if (delta0 < delta1) {
            if (search_left) {
                knn_search_helper_v1(node->right, query, heap, k, threads_left);
            }
            else {
                knn_search_helper_v1(node->left, query, heap, k, threads_left);
            }
        }
    }

    // #pragma region O0
    //     if (threads_left <= 1) {
    //         // Search left subtree
    //         if (search_left) {
    //             knn_search_helper_v1(node->left, query, heap, k, threads_left);
    //         }
    //         // Search right subtree
    //         else {
    //             knn_search_helper_v1(node->right, query, heap, k, threads_left);
    //         }

    //         if ((int)heap.size() < k) {
    //             if (search_left) {
    //                 knn_search_helper_v1(node->right, query, heap, k, threads_left);
    //             }
    //             else {
    //                 knn_search_helper_v1(node->left, query, heap, k, threads_left);
    //             }
    //         }
    //         else {
    //             auto first = heap.front();
    //             double delta0 = std::fabs(node->point[axis] - query[axis]);
    //             double delta1 = std::fabs(first[axis] - query[axis]);
    //             if (delta0 < delta1) {
    //                 if (search_left) {
    //                     knn_search_helper_v1(node->right, query, heap, k, threads_left);
    //                 }
    //                 else {
    //                     knn_search_helper_v1(node->left, query, heap, k, threads_left);
    //                 }
    //             }
    //         }
    //     } else {
    //         int l_threads {}, r_threads {};
    //         if (search_left) {
    //             r_threads = threads_left / 2;
    //             l_threads = threads_left - r_threads;
    //         } else {
    //             l_threads = threads_left / 2;
    //             r_threads = threads_left - l_threads;
    //         }
    //         #pragma omp task shared(node, query, heap, k) private(l_threads)
    //         knn_search_helper_v1(node->left, query, heap, k, l_threads);
    //         #pragma omp task shared(node, query, heap, k) private(r_threads)
    //         knn_search_helper_v1(node->left, query, heap, k, r_threads);
    //         // #pragma omp taskwait
    //     }
    // #pragma endregion
}

template class SkdTree<Point2D>;
template class SkdTree<Point3D>;
template class PkdTree<Point2D>;
template class PkdTree<Point3D>;