#pragma once

#include <cmath>
#include <eigen3/Eigen/Eigen>

namespace linear_system
{

/**
 * @brief NchooseK Combination of N items taken K at a time.
 * @param N number of choices
 * @param K number of selected choices
 * @return The binomial coefficient
 */
unsigned int NchooseK(unsigned int N, unsigned int K);

/**
 * @brief PolynomialDegree The order of a polynomial.
 * @param P the polynomial
 * @return The order
 */
template<typename Derived>
unsigned int PolynomialDegree(const Eigen::MatrixBase<Derived> & P)
{
    int ret = 0;
    for (unsigned int i = 0; i < P.size(); i++)
    {
        if (std::fabs(P(i)) > 1e-8)
        {
            ret = P.size() - i - 1;
            break;
        }
    }
    return ret;
}

template<typename Derived>
Eigen::VectorXd VectorShiftLeft(const Eigen::MatrixBase<Derived> & V, int n)
{
    int s = V.size();
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(s);
    ret.head(s-n) = V.tail(s-n);
    return ret;
}

/*!
 * \brief PolynomialDivision Performs a polynomial division such that N/D = q/D + r
 *
 * Note that q and r should have the same size as D prior to calling this function.
 *
 * \param N Numerator
 * \param D Denominator
 * \param q Quotient
 * \param r Remainder
 */
template<typename Derived>
void PolynomialDivision(const Eigen::MatrixBase<Derived> & N, const Eigen::MatrixBase<Derived> & D, Eigen::MatrixBase<Derived> & q, Eigen::MatrixBase<Derived> & r)
{
    unsigned int s = D.size();

    // These instructions only assert that the new size equals the old size, and do nothing else
    // This is so because N, D are MatrixBase objects
    q.resize(s);
    r.resize(s);

    q.setZero();
    r.setZero();
    unsigned int i = 0;

    Eigen::VectorXd N_aux(s);
    N_aux = N;
    Eigen::VectorXd D_aux(s);
    int degD = PolynomialDegree(D);
    int degN_aux = PolynomialDegree(N_aux);
    int n = degN_aux - degD;
    double a;

    while (n >= 0)
    {
        D_aux = VectorShiftLeft(D,n);
        a = N_aux(i)/D_aux(i);
        q(s-n-1) = a;
        D_aux *= a;
        N_aux -= D_aux;
        i ++;
        n = PolynomialDegree(N_aux) - degD;
    }
    r = N_aux;
}

/*!
 * \brief Wraps the angle to the (-pi,pi] interval
 *
 * \param ang angle to be wrapped
 */
void wrap2pi(double &ang);
void wrap2pi(Eigen::VectorXd &ang);

}
