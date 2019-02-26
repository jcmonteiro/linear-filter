#pragma once

#include <Eigen/Eigen>
#include <stdint.h>
#include <stdexcept>

namespace linear_system
{

enum IntegrationMethod
{
    FORWARD_EULER,
    BACKWARD_EULER,
    TUSTIN
};


/*!
 * \brief The LinearSystem class implements multiple identical N-th order linear filters at once
 */
class LinearSystem
{
public:
    typedef int64_t Time;

    /*! @brief Default maximum time (in seconds) that is allowed to pass between calls to Update */
    const static double DEFAULT_MAX_TIME_BETWEEN_UPDATES;

    static Time TimeFromSeconds(double time);

private:
    //SS realization
    Eigen::MatrixXd A;
    Eigen::VectorXd B;
    Eigen::RowVectorXd C;
    double D;

    unsigned int order;
    Eigen::MatrixXd state; //States of the state-space in matrix form (numInputs,stateSize)

    /*! @brief Filter numerator tfNum[0] s^N + tfNum[1] s^(N-1) + ... + tfNum[N] */
    Eigen::VectorXd tfNum;

    /*! @brief Filter denominator tfDen[0] s^N + tfDen[1] s^(N-1) + ... + tfDen[N] */
    Eigen::VectorXd tfDen;

    /*! @brief Sampling period (in seconds) */
    double Ts;

    /*! @brief Amount of filters */
    unsigned int n_filters;

    /*! @brief Filter current time */
    Time time_current;

    /*! @brief Indicates whether or not the initial time has been set */
    bool time_init_set;

    /*! @brief Last output value */
    Eigen::VectorXd last_output;

    /*!
     * \brief Initial output and its N-1 derivatives for each filter
     *
     * Each row contains the j-th filter output and its N-1 derivatives at time t = 0
     */
    Eigen::MatrixXd initialOutputDerivatives;

    /*! @brief Maximum amount time between successive calls to Update */
    Time max_delta;

    /*! @brief Integration method */
    IntegrationMethod integration_method;

    /*!
     * @brief Prewarp frequency
     *
     * Frequency (in rad/s) for which the magnitude response does not change when going
     * from continuous to discrete-time. This is only considered when using Tustin
     * integration method.
     */
    double prewarp_frequency;

    void ConvertFwdEuler(Eigen::VectorXd & poly) const;
    void ConvertBwdEuler(Eigen::VectorXd & poly) const;
    void ConvertTustin(Eigen::VectorXd & poly) const;

    /*!
     * \brief Update Updates all filters (one sample period) based on the given inputs
     * \param signalIn input signals
     */
    void Update(const Eigen::RowVectorXd &signalIn);

    /*!
     * \brief TF2SS Computes the state-space realization (A,B,C,D)
     */
    void TF2SS();

public:
    LinearSystem(Eigen::VectorXd _tfNum = Eigen::VectorXd::Zero(2), Eigen::VectorXd _tfDen = Eigen::VectorXd::Constant(2,1),
                     double _Ts = 0.001, IntegrationMethod _integrationMethod = TUSTIN, double _prewarpFrequency = 0);

    /*!
     * \brief setIntegrationMethod Configures the integration method
     * \param method The integration method
     */
    inline void setIntegrationMethod(IntegrationMethod method) {integration_method = method;}

    /*!
     * \brief getIntegrationMethod Returns the integration method
     * \return The integration method
     */
    inline IntegrationMethod getIntegrationMethod() {return integration_method;}

    /*!
     * \brief setPrewarpFrequency Configures the prewarp frequency used with Tustin's integration method
     * \param frequency The prewarp frequency
     */
    inline void setPrewarpFrequency(double frequency)
    {
        if (frequency >= 0)
            prewarp_frequency = frequency;
        else
            throw std::invalid_argument("LinearSystem::setPrewarpFrequency - frequency must be nonnegative");
    }

    /*!
     * \brief getPrewarpFrequency Returns the prewarp frequency used with Tustin's integration method
     * \return The prewarp frequency
     */
    inline double getPrewarpFrequency() {return prewarp_frequency;}

    /*!
     * \brief GetOrder Returns the filter order
     * \return The filter order
     */
    inline unsigned int GetOrder() const
    {
        return tfDen.size() - 1;
    }

    /*!
     * \brief GetCoefficients Returns the filters coefficients
     * \param coef_num Numerator coefficients
     * \param coef_den Denominator coefficients
     */
    inline void GetCoefficients(Eigen::VectorXd & coef_num, Eigen::VectorXd & coef_den) const
    {
        coef_num = tfNum;
        coef_den = tfDen;
    }

    /*!
     * \brief UseNFilters Chooses how many filters should run in parallel
     */
    void UseNFilters(const unsigned int n_filters);

    /*!
     * \brief SetFilter Configures the numerator and denominator used by the filters
     * \param coef_num Numerator coefficients coef_num[0] s^N + coef_num[1] s^(N-1) + ... + coef_num[N]
     * \param coef_den Denominator coefficients coef_den[0] s^N + coef_den[1] s^(N-1) + ... + coef_den[N]
     */
    void SetFilter(const Eigen::VectorXd & coef_num, const Eigen::VectorXd & coef_den);

    /*!
     * \brief SetSampling Sets the sampling period (in seconds)
     *
     * A call to this method silently changes the maximum allowed time between updates
     * if the new sampling period >= GetMaximumTimeBetweenUpdates(). When this happens,
     * this methods makes a call to SetMaximumTimeBetweenUpdates(10 * sampling_period).
     *
     * \param sampling_period positive sampling period
     */
    void SetSampling(const double sampling_period);

    /**
     * @brief GetSampling Returns the sampling period.
     * @return The sampling period.
     */
    inline double GetSampling() const {return Ts;}

    /*!
     * \brief GetMaximumTimeBetweenUpdates Returns the maximum time (in seconds) between calls to Update
     * \return The maximum time between updates
     */
    inline double GetMaximumTimeBetweenUpdates() const {return ((double) max_delta) / 1000000;}

    /*!
     * \brief GetNFilters Returns the number of filters
     * \return The number of filters
     */
    inline unsigned int GetNFilters() const {return n_filters;}

    inline const Eigen::VectorXd & GetOutput() {return last_output;}

    /*!
     * \brief SetMaximumTimeBetweenUpdates Sets the maximum time (in seconds) between calls to Update
     * \param delta_time The maximum time between updates
     */
    void SetMaximumTimeBetweenUpdates(double delta_time);

    /*!
     * \brief DiscretizeSystem Transforms the filter to discrete time
     */
    void DiscretizeSystem();

    inline void SetInitialTime(Time time) {time_current = time; time_init_set = true;}

    /*!
     * \brief Update Updates all filters based on the given inputs until they reach the current time
     * \param signalIn input signals
     * \param time current time (in microseconds)
     * \return The output of every filter
     */
    Eigen::VectorXd Update(const Eigen::RowVectorXd &signalIn, Time time);

    void SetState(Eigen::MatrixXd _state){state = _state;}
    inline Eigen::MatrixXd GetState() const {return state;}

    /*!
     * \brief SetInitialState Sets the initial state x[0] of the N-th order filter
     *
     * The initial state is computed from:
     *     u[0], ..., u[-(N-1)], the current and N-1 past input samples
     *     y[0], dy/dt[0], ..., d^(N-1)y/dt^(N-1)[0], the current output and its N-1 derivatives
     *
     * The output and its derivatives must be set via SetInitialOutputDerivatives
     *
     * \param u_history each i-th row holds u_i[0], ..., u_i[-(N-1)], in this order
     * \see SetInitialOutputDerivatives
     */
    void SetInitialState(const Eigen::MatrixXd & u_history);

    /*!
     * \brief SetInitialState Sets the initial state x[0] of the 1-st order filter
     *
     * This method should only be called when the filter is of 1-st order
     *
     * \param u_channels holds u_1[0], u_2[0], ..., u_M[0], where M is the number of filters
     * \see UseNFilters
     */
    void SetInitialState(const Eigen::VectorXd & u_channels);

    /*!
     * \brief SetInitialOutputDerivatives Sets the initial output and its N-1 derivatives,
     * where N is the filter order, for each channel
     *
     * Every row contains y[0], dy/dt[0], ..., d^(N-1)y/dt^(N-1)[0], in this order and each row
     * corresponds to one independent instance of the filter defined by this class
     *
     * \param _initialOutputDerivatives
     */
    void SetInitialOutputDerivatives(const Eigen::MatrixXd & _initialOutputDerivatives);
};

}
