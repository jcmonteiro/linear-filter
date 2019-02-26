#include "LinearSystem.hpp"
#include "HelperFunctions.hpp"
#include <cmath>
#include <iostream>

using namespace linear_system;

const double LinearSystem::DEFAULT_MAX_TIME_BETWEEN_UPDATES = 1;

LinearSystem::LinearSystem(Eigen::VectorXd _tfNum, Eigen::VectorXd _tfDen, double _Ts, IntegrationMethod _integrationMethod, double _prewarpFrequency) :
    n_filters(1), time_current(0), time_init_set(false)
{
    this->SetFilter(_tfNum, _tfDen);
    this->UseNFilters(1);
    this->SetSampling(_Ts);
    this->SetMaximumTimeBetweenUpdates(DEFAULT_MAX_TIME_BETWEEN_UPDATES);

    integration_method = _integrationMethod;
    prewarp_frequency = _prewarpFrequency;
}

void LinearSystem::UseNFilters(const unsigned int n_filters)
{
    if (n_filters == 0)
        throw std::logic_error("received n_filters = 0, but LinearSystem must implement at least one filter");

    initialOutputDerivatives.setZero(n_filters, order);
    state.setZero(n_filters, order);
    this->n_filters = n_filters;
    last_output.setZero(n_filters);
}

void LinearSystem::SetFilter(const Eigen::VectorXd & coef_num, const Eigen::VectorXd & coef_den)
{
    if (coef_den.size() == 0)
        throw std::logic_error("invalid system order, since there are no denominator coefficients to set");

    if (coef_num.size() > coef_den.size())
        throw std::logic_error("the numerator order should not be higher than the denominator order");

    if (coef_den(0) == 0)
        throw std::logic_error("denominator's first term can't be zero");

    tfDen = coef_den;

    // Make sure the numerator has as many coefficients as the denominator, even
    // if some of the higher terms are zero
    if (coef_den.size() > coef_num.size())
    {
        Eigen::VectorXd tmp_num(tfDen.size());
        tmp_num << Eigen::VectorXd::Zero(coef_den.size() - coef_num.size(), 1), coef_num;
        tfNum = tmp_num;
    }
    else
        tfNum = coef_num;

    // Normalize vectors such that the denominator is monic
    tfNum = tfNum / tfDen(0);
    tfDen = tfDen / tfDen(0);

    // Set the filter order
    order = tfDen.size() - 1;

    // Reset state matrices
    A.setZero(order,order);
    B.setZero(order);
    C.setZero(order);
    D = 0;
    //
    state.setZero(n_filters, order);

    // Reset output derivatives matrix
    initialOutputDerivatives.setZero(n_filters, order);

    // Reset last output
    last_output.setZero(n_filters);
}

void LinearSystem::SetSampling(const double sampling_period)
{
    if (sampling_period <= 0.0)
        throw std::logic_error("non positive sampling time given");

    Ts = sampling_period;

    if (Ts >= GetMaximumTimeBetweenUpdates())
        SetMaximumTimeBetweenUpdates(10 * Ts);
}

void LinearSystem::SetMaximumTimeBetweenUpdates(double delta_time)
{
    if (delta_time <= 0.0)
        throw std::logic_error("non positive time given");

    max_delta = 1000000L * delta_time;
}

void LinearSystem::ConvertFwdEuler(Eigen::VectorXd & poly) const
{
    Eigen::VectorXd poly_old = poly;
    poly.setZero();
    //
    for (unsigned int k = 0; k <= order; ++k)
    {
        for (unsigned int j = k; j <= order; ++j)
            poly(order-k) += NchooseK(j,k) * std::pow(-1,j-k) * poly_old(order-j) * std::pow(Ts,order-j);
    }
}

void LinearSystem::ConvertBwdEuler(Eigen::VectorXd & poly) const
{
    Eigen::VectorXd poly_old = poly;
    poly.setZero();
    //
    for (unsigned int k = 0; k <= order; k++)
    {
        for (unsigned int j = k; j <= order; j++)
            poly(k) += NchooseK(j,k) * std::pow(-1,k) * poly_old(order-j) * std::pow(Ts,order-j);
    }
}

void LinearSystem::ConvertTustin(Eigen::VectorXd & poly) const
{
    Eigen::VectorXd poly_old = poly;
    poly.setZero();
    //
    Eigen::VectorXd tustin_sum(order + 1);
    Eigen::VectorXd tustin_coefs(order + 1);
    double tustin_a = (prewarp_frequency != 0) ? prewarp_frequency / tan(prewarp_frequency * Ts / 2) : 2 / Ts;
    //
    for (unsigned int k = 0; k <= order; k++)
    {
        tustin_sum.setZero();
        for (unsigned int j = 0; j <= k; j++)
        {
            tustin_coefs.setZero();
            for (unsigned int i = 0; i <= (order - k); i++) {
                tustin_coefs(i+j) = NchooseK(order-k,i);
            }
            tustin_sum += NchooseK(k,j) * std::pow(-1,j) * tustin_coefs;
        }
        poly += std::pow(tustin_a,k) * poly_old(order-k) * tustin_sum;
    }
}

void LinearSystem::DiscretizeSystem()
{
    switch(integration_method)
    {
    case FORWARD_EULER:
        this->ConvertFwdEuler(tfNum);
        this->ConvertFwdEuler(tfDen);
        break;
    case BACKWARD_EULER:
        this->ConvertBwdEuler(tfNum);
        this->ConvertBwdEuler(tfDen);
        break;
    case TUSTIN:
        this->ConvertTustin(tfNum);
        this->ConvertTustin(tfDen);
        break;
    default: throw std::logic_error("invalid integration method");
    }
    tfNum /= tfDen(0);
    tfDen /= tfDen(0);
    TF2SS();
}

void LinearSystem::TF2SS()
{
    if (order <= 0)
        throw std::logic_error("invalid system order");

    Eigen::VectorXd num(order + 1);

    if (PolynomialDegree(tfNum) == PolynomialDegree(tfDen))
    {
        Eigen::VectorXd quotient(order + 1);
        PolynomialDivision(tfNum, tfDen, quotient, num);
        D = quotient(order);
    }
    else
        num = tfNum;

    A.setZero();
    A.topRightCorner(order-1, order-1) = Eigen::MatrixXd::Identity(order-1, order-1);
    for (unsigned int i = 0; i < order; i++)
        A(order-1,i) = -tfDen(order-i);

    B.setZero();
    B(order-1) = 1;

    C.setZero();
    for (unsigned int i = 0; i < order; i++)
        C(i) = num(order-i);
}

Eigen::VectorXd LinearSystem::Update(const Eigen::RowVectorXd &signalIn, Time time)
{
    Time delta = time - time_current;
    if (!time_init_set)
    {
        std::cerr << "[WARN] (LinearSystem) The filter initial time is not set! Returning zero!" << std::endl;
        return Eigen::VectorXd::Zero(n_filters);
    }
    else if (delta < 0)
    {
        std::fprintf(stderr, "[WARN] (LinearSystem) The requested update requires a trip to the past, filter time (%ld) > time asked (%ld)",
                     time_current, time);
        std::cerr << ", the output is set to its previous value (the initial one if it was never updated). Are you providing the time in microseconds?"
                  << std::endl;
        return last_output;
    }
    else if (delta > max_delta)
    {
        std::fprintf(stderr, "[WARN] (LinearSystem) There has been a long time since the last update (%.3f seconds)", GetMaximumTimeBetweenUpdates());
        std::cerr << ". The filter will reset its state (based on the current input) to match the last output. If this is not acceptable, "
                  << "adjust the maximum update time in setMaximumUpdateTime." << std::endl;
        Eigen::MatrixXd u_history(n_filters, order), ydy(n_filters, order);
        //
        for (unsigned int k = 0; k < order; ++k)
            u_history.col(k) = signalIn;
        //
        ydy.setZero();
        ydy.col(0) = last_output;
        //
        SetInitialOutputDerivatives(ydy);
        SetInitialState(u_history);
        SetInitialTime(time);
        time_current = time;
        return last_output;
    }


    Time iterations = std::roundl( (delta / GetSampling()) / 1000000L );

    if (iterations == 0)
        return last_output;

    time_current = time;

    for (unsigned int k = 1; k < iterations; ++k)
    {
        Update(signalIn);
    }
    Update(signalIn);
    return last_output;
}

void LinearSystem::Update(const Eigen::RowVectorXd &signalIn)
{
    if (signalIn.size() != n_filters)
    {
        if (signalIn.size() > n_filters)
            throw std::logic_error("there are more inputs than filters");
        else
            throw std::logic_error("there are less inputs than filters");
    }

    last_output = C * state.transpose() + D * signalIn;

    state = A * state.transpose() + B * signalIn;
    state.transposeInPlace();
}

void LinearSystem::SetInitialState(const Eigen::MatrixXd & u_history)
{
    if (u_history.cols() != order)
    {
        char buffer[70];
        std::sprintf(buffer, "expected %d input %s per row, but received %d",
                     order,
                     (order == 1) ? "entry" : "entries",
                     (int) u_history.cols());
        throw std::logic_error(buffer);
    }

    if (u_history.rows() != n_filters)
    {
        throw std::logic_error("the number of input channels is different from the number of filters");
    }

    Eigen::MatrixXd Cbar(order,order);
    Eigen::MatrixXd Dbar(order,order);
    Eigen::VectorXd y(order);
    Eigen::RowVectorXd tmp(order);
    double acc;

    // Compute initial condition for each filter
    for (unsigned int i = 0; i < state.rows(); i++)
    {
        Cbar.setZero();
        Cbar.row(0) = C;
        Dbar.setZero();
        y.setZero();
        y(0) = initialOutputDerivatives(i,0);

        for (unsigned int j = 1; j < order; j++)
        {
            tmp = A.transpose().colPivHouseholderQr().solve(Cbar.row(j-1).transpose()); // equivalent to tmp = Cbar(j-1,:) / A
            Cbar.row(j) = tmp;
            Dbar(j,1) = tmp * B;

            if (j > 1)
                Dbar.block(j,2,1,j-1) = Dbar.block(j-1,1,1,j-1);

            acc = 0;
            for (unsigned int k = 0; k <= j-1; k++)
                acc += NchooseK(j,k) * std::pow(-1,k) * y(k);

            y(j) = (std::pow(Ts,j) * initialOutputDerivatives(i,j) - acc) * std::pow(-1,j);
        }
        Dbar -= D*Eigen::MatrixXd::Identity(order,order);
        state.row(i) = Cbar.colPivHouseholderQr().solve(Dbar * u_history.row(i).transpose() + y);
    }
}

void LinearSystem::SetInitialState(const Eigen::VectorXd & u_channels)
{
    if (order > 1)
        throw std::logic_error("SetInitialState cannot be called with a vector as input if the filter order is greater than 1");
    if (n_filters != u_channels.size())
    {
        char buffer[100];
        std::sprintf(buffer, "expected %d input %s, one for each filter, but received %d",
                     n_filters,
                     (n_filters == 1) ? "entry" : "entries",
                     (int) u_channels.size());
        throw std::logic_error(buffer);
    }
    Eigen::MatrixXd u_history(n_filters, 1);
    u_history.col(0) = u_channels;
    this->SetInitialState(u_history);
}

void LinearSystem::SetInitialOutputDerivatives(const Eigen::MatrixXd & _initialOutputDerivatives)
{
    if (_initialOutputDerivatives.cols() != order)
    {
        char buffer[70];
        std::sprintf(buffer, "expected %d %s per row, but received %d",
                     order,
                     (order == 1) ? "element" : "elements",
                     (int) _initialOutputDerivatives.cols());
        throw std::logic_error(buffer);
    }

    if (_initialOutputDerivatives.rows() != n_filters)
    {
        char buffer[70];
        std::sprintf(buffer, "expected %d %s per column, but received %d",
                     n_filters,
                     (n_filters == 1) ? "element" : "elements",
                     (int) _initialOutputDerivatives.rows());
        throw std::logic_error(buffer);
    }

    initialOutputDerivatives = _initialOutputDerivatives;
    last_output = initialOutputDerivatives.col(0);
}

LinearSystem::Time LinearSystem::TimeFromSeconds(double time)
{
    return 1000000L * time;
}
