#ifndef XALODE_UTILS_H
#define XALODE_UTILS_H

// stl
#include <algorithm>
#include <vector>


struct target_less
{
    template<class It>
    bool operator()(It const &a, It const &b) const { return *a < *b; }
};


struct target_equal
{
    template<class It>
    bool operator()(It const &a, It const &b) const { return *a == *b; }
};


template<class T>
T uniquify(T begin, T const end)
{
    std::vector<T> v;
    v.reserve(static_cast<size_t>(std::distance(begin, end)));
    for (T i = begin; i != end; ++i)
        v.emplace_back(i);

    std::sort(v.begin(), v.end(), target_less());

    v.erase(std::unique(v.begin(), v.end(), target_equal()), v.end());
    std::sort(v.begin(), v.end());
    size_t j = 0;
    for (T i = begin; i != end && j != v.size(); ++i)
    {
        if (i == v[j])
        {
            std::iter_swap(i, begin);
            ++j;
            ++begin;
        }
    }
    return begin;
}

#endif
