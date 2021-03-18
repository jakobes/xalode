#ifndef UTILS_H
#define UTILS_H


#include <algorithm>
#include <functional>


template< class vector_type, typename float_type = typename vector_type::value_type >
void axpy(vector_type &x, vector_type &y, const float_type a)
{
    std::transform(x.begin(), x.end(), y.begin(), x.begin(),
            [a](const float_type xi, const float_type yi){return a*xi + yi;});
}


template< class vector_type, typename float_type = typename vector_type::value_type >
void multiply(vector_type &x, const float_type a)
{
    std::transform(x.begin(), x.end(), x.begin(), [a](float_type xi){return a*xi;});
}



#endif
