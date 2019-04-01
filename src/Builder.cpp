#include "Builder.hpp"
#include "HelperFunctions.hpp"


using namespace linear_system;


LinearSystem Builder::createSecondOrder(double damp, double cutoff)
{
    Poly num(3), den(3);
    double wn = cutoff2resonant(cutoff, damp);
    num << 0, wn*wn, 0;
    den << 1, 2*damp*wn, wn*wn;
    return LinearSystem(num, den);
}

LinearSystem Builder::createReferenceFilter2I(double kp, double ki, double kd)
{
    Poly num(1), den(3);
    num << ki;
    den << kd, kp, ki;
    return LinearSystem(num, den);
}

LinearSystem Builder::createReferenceFilter2I(double kp, double kd)
{
    Poly num(1), den(2);
    num << kp;
    den << kd, kp;
    return LinearSystem(num, den);
}

LinearSystem Builder::createReferenceFilterI(double kp, double ki)
{
    Poly num(1), den(2);
    num << ki;
    den << kp, ki;
    return LinearSystem(num, den);
}
