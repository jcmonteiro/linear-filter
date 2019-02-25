#include "HelperFunctions.hpp"

unsigned int linear_system::NchooseK(unsigned int N, unsigned int K)
{
    double ret = 1;
    for (unsigned int i = 1; i <= K; i++)
        ret *= (N + 1 - i)/(double)i;
    return (unsigned int)ret;
}
