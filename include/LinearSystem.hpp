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

typedef int64_t Time;
typedef Eigen::VectorXd Poly;
typedef Eigen::RowVectorXd Input;
typedef Eigen::VectorXd Output;

/*!
 * \brief The LinearSystem class implements multiple identical N-th order linear filters at once
 */
class LinearSystem
{
public:
    static Time getTimeFromSeconds(double time);

private:
    //SS realization
    Eigen::MatrixXd A;
    Eigen::VectorXd B;
    Eigen::RowVectorXd C;
    double D;

    unsigned int order;
    Eigen::MatrixXd state; //States of the state-space in matrix form (numInputs,stateSize)

    /*! @brief Filter numerator tfNum[0] s^N + tfNum[1] s^(N-1) + ... + tfNum[N] */
    Poly tf_num;

    /*! @brief Filter denominator tfDen[0] s^N + tfDen[1] s^(N-1) + ... + tfDen[N] */
    Poly tf_den;

    /*! @brief Sampling period (in seconds) */
    double Ts;

    /*! @brief Amount of filters */
    unsigned int n_filters;

    /*! @brief Filter current time */
    Time time_current;

    /*! @brief Indicates whether or not the initial time has been set */
    bool time_init_set;

    /*! @brief Last output value */
    Output last_output;

    /*!
     * \brief Initial output and its N-1 derivatives for each filter
     *
     * Each row contains the j-th filter output and its N-1 derivatives at time t = 0
     */
    Eigen::MatrixXd initial_output_derivatives;

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

    /*!
     * \brief Transforms the filter to discrete time.
     */
    void discretize();

    /**
     * @brief Converts the polynomial \p poly from continuous-time do discrete-time using
     * the forward Euler approximation
     */
    void convertFwdEuler(Poly & poly) const;

    /**
     * @brief Converts the polynomial \p poly from continuous-time do discrete-time using
     * the backward Euler approximation
     */
    void convertBwdEuler(Poly & poly) const;

    /**
     * @brief Converts the polynomial \p poly from continuous-time do discrete-time using
     * the Tustin approximation
     */
    void convertTustin(Poly & poly) const;

    /*!
     * \brief update Updates all filters (one sample period) based on the given inputs
     * \param signalIn input signals
     */
    void update(const Input &signalIn);

    /*!
     * \brief Computes the state-space realization (A,B,C,D)
     */
    void tf2ss();

    /*!
     * \brief setFilter Configures the numerator and denominator used by the filters
     * \param coef_num Numerator coefficients coef_num[0] s^N + coef_num[1] s^(N-1) + ... + coef_num[N]
     * \param coef_den Denominator coefficients coef_den[0] s^N + coef_den[1] s^(N-1) + ... + coef_den[N]
     * \param
     */
    void setFilter(const Poly &coef_num, const Poly &coef_den);

    /*!
     * \brief setInitialState Sets the initial state x[0] of the N-th order filter
     *
     * The initial state is computed from:
     *     u[0], ..., u[-(N-1)], the current and N-1 past input samples
     *     y[0], dy/dt[0], ..., d^(N-1)y/dt^(N-1)[0], the current output and its N-1 derivatives
     *
     * The output and its derivatives must be set via #setInitialOutputDerivatives
     *
     * \param u_history each i-th row holds u_i[0], ..., u_i[-(N-1)], in this order; that is, the
     * current input and last N-1 inputs of the i-th input channel
     * \see #setInitialOutputDerivatives
     */
    void setInitialState(const Eigen::MatrixXd & u_history);

    /*!
     * \brief setInitialOutputDerivatives Sets the initial output and its N-1 derivatives,
     * where N is the filter order, for each channel
     *
     * \param initial_output_derivatives Every row contains y[0], dy/dt[0], ...,
     * d^(N-1)y/dt^(N-1)[0], in this order and each row corresponds to one independent
     * instance of the filter defined by this class
     *
     * \see #setInitialState
     */
    void setInitialOutputDerivatives(const Eigen::MatrixXd & initial_output_derivatives);

    /*!
     * \brief Configures the prewarp frequency used with Tustin's integration method.
     *
     * This method must be called before setFilter.
     *
     * \param frequency The prewarp frequency
     */
    inline void setPrewarpFrequency(double frequency)
    {
        if (integration_method != TUSTIN)
            frequency = 0;
        if (frequency >= 0)
            prewarp_frequency = frequency;
        else
            throw std::invalid_argument("LinearSystem::setPrewarpFrequency - frequency must be nonnegative");
    }

    /*!
     * \brief Sets the sampling period (in seconds).
     *
     * A call to this method silently changes the maximum allowed time between updates
     * if the new sampling period >= #getMaximumTimeBetweenUpdates. When this happens,
     * this methods makes a call to #setMaximumTimeBetweenUpdates(10 * sampling_period).
     *
     * \param sampling_period positive sampling period.
     */
    void setSampling(double sampling_period);

public:
    /**
     * @brief Constructor.
     * @param num Filter numerator.
     * @param den Filter denominator.
     * @param ts Filter sampling time.
     * @param method Integration method.
     * @param prewarp Prewarp frequency to use with Tustin's integration method. Use 0 to
     * disable it. Defaults to 0.
     */
    LinearSystem(Poly num = Poly::Zero(1), Poly den = Poly::Constant(1,1), double ts = 0.001,
        IntegrationMethod method = TUSTIN, double prewarp = 0);

    /*!
     * \brief Returns the integration method chosen when calling setFilter;
     * defaults to #IntegrationMethod::Tustin
     * \return The integration method
     */
    inline IntegrationMethod getIntegrationMethod() const {return integration_method;}

    /*!
     * \brief Returns the prewarp frequency used with Tustin's integration method.
     * \return The prewarp frequency.
     */
    inline double getPrewarpFrequency() const {return prewarp_frequency;}

    /*!
     * \brief getOrder Returns the filter order
     * \return The filter order
     */
    inline unsigned int getOrder() const
    {
        return tf_den.size() - 1;
    }

    /*!
     * \brief getCoefficients Returns the filters coefficients
     * \param coef_num Numerator coefficients
     * \param coef_den Denominator coefficients
     */
    inline void getCoefficients(Eigen::VectorXd & coef_num, Eigen::VectorXd & coef_den) const
    {
        coef_num = tf_num;
        coef_den = tf_den;
    }

    /*!
     * \brief Chooses how many filters should run in parallel.
     *
     * This method must be called before #setInitialConditions
     */
    void useNFilters(unsigned int n_filters);

    /**
     * @brief Configures the filter initial state given the current input and its N-1 previous
     * values and the desired initial output and its N-1 derivatives.
     *
     * One must still call #setInitialTime after (or before) calling this method.
     *
     * @param init_in A (#getNFilters by #getOrder) matrix where each row contains u_i[0], ..., u_i[-(N-1)],
     * the current and N-1 past input samples of the i-th input channel.
     * @param init_out_dout A (#getNFilters by #getOrder) matrix where each row contains y_i[0], dy_i/dt[0],
     * ..., d^(N-1)y_i/dt^(N-1)[0], the i-th output and its N-1 derivatives.
     *
     * \see setInitialTime
     */
    void setInitialConditions(const Eigen::MatrixXd &init_in, const Eigen::MatrixXd &init_out_dout);

    /*!
     * \brief setInitialTime Sets the filter initial time.
     * \param time The initial time.
     * \see setInitialConditions
     */
    inline void setInitialTime(Time time) {time_current = time; time_init_set = true;}

    /**
     * @brief Returns the sampling period in seconds.
     * @return The sampling period.
     */
    inline double getSampling() const {return Ts;}

    /**
     * @brief Returns the sampling period in microseconds.
     * @return The sampling period.
     */
    inline Time getSamplingMicro() const {return Ts * 1000000L;}

    /*!
     * \brief Returns the maximum time (in seconds) between calls to #update.
     * \return The maximum time between updates.
     */
    inline double getMaximumTimeBetweenUpdates() const {return ((double) max_delta) / 1000000;}

    /*!
     * \brief Returns the number of filters.
     * \return The number of filters.
     */
    inline unsigned int getNFilters() const {return n_filters;}

    /**
     * @brief Returns the last output returned by this filter.
     */
    inline const Output & getOutput() const {return last_output;}

    /*!
     * \brief Sets the maximum time (in seconds) between calls to #update
     * \param delta_time The maximum time between updates.
     */
    void setMaximumTimeBetweenUpdates(double delta_time);

    /*!
     * \brief Updates all filters based on the given inputs until they reach the current time.
     * \param signalIn input signals.
     * \param time current time (in microseconds).
     * \return The output of every filter.
     */
    Output update(const Input &signalIn, Time time);

    /**
     * @brief Forces a state for each filter.
     * @param state A (#getNFilters by #getOrder) matrix where each row holds
     * the state of the i-th filter.
     */
    void setState(const Eigen::MatrixXd &state){this->state = state;}

    /**
     * @brief Returns the states of each one of the #getNFilters filters
     * @return A (#getNFilters by #getOrder) matrix where each row holds
     * the state of the i-th filter.
     */
    inline Eigen::MatrixXd getState() const {return state;}
};

}
