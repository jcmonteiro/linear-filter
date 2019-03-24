#pragma once


#include "LinearSystem.hpp"


namespace linear_system
{


/**
 * @brief Helper class to build some common linear filters.
 */
class Builder
{
public:
    /**
     * @brief Returns a second order filter with prescribed damping and cutoff frequency.
     * @param damp Damping coefficient.
     * @param cutoff Cutoff frequency.
     * @return The second order filter.
     */
    static LinearSystem createSecondOrder(double damp, double cutoff);

    /**
     * @brief Returns a reference filter considering PID control of a double integrator.
     * @param kp Proportional gain.
     * @param ki Integral gain.
     * @param kd Derivative gain.
     * @return The reference filter.
     */
    static LinearSystem createReferenceFilter2I(double kp, double ki, double kd);

    /**
     * @brief Returns a reference filter considering PD control of a double integrator.
     * @param kp Proportional gain.
     * @param kd Derivative gain.
     * @return The reference filter.
     */
    static LinearSystem createReferenceFilter2I(double kp, double kd);

    /**
     * @brief Returns a reference filter considering PI control of an integrator.
     * @param kp Proportional gain.
     * @param ki Integral gain.
     * @return The reference filter.
     */
    static LinearSystem createReferenceFilterI(double kp, double ki);
};


}
