#define BOOST_TEST_MODULE LinearSystem test

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <HelperFunctions.hpp>
#include <LinearSystem.hpp>
#include <limits>

using namespace linear_system;

template<int I, int J>
void operator >> (const YAML::Node & node, Eigen::Matrix<double, I, J> & m)
{
    auto r = m.rows();
    auto c = m.cols();

    // the yml data is written column by column
    for (unsigned int j = 0; j < c; j++)
    {
        for (unsigned int i = 0; i < r; i++)
        {
            unsigned int index = r*j + i;
            m(i, j) = node[index].as<double>();
        }
    }
}

void GetYmlData(const YAML::Node & node, int & n, int & order, LinearSystem & ls, Eigen::VectorXd & u, Eigen::VectorXd & y_tustin, Eigen::VectorXd & y_fwd, Eigen::VectorXd & y_bwd, double & Ts)
{
    //Data parameters
    double omega;
    n = node["n"].as<int>();
    order = node["order"].as<int>();
    Ts = node["Ts"].as<double>();
    omega = node["omega"].as<double>();\

    //Data
    u.resize(n);
    y_tustin.resize(n);
    y_fwd.resize(n);
    y_bwd.resize(n);
    node["u"] >> u;
    node["y_tustin"] >> y_tustin;
    node["y_fwd"] >> y_fwd;
    node["y_bwd"] >> y_bwd;

    Poly num(order+1);
    Poly den(order+1);
    Eigen::MatrixXd ydy0(1,order);
    node["num"] >> num;
    node["den"] >> den;
    node["ydy0"] >> ydy0;

    ls.setSampling(Ts);
    ls.setPrewarpFrequency(omega);
    ls.setFilter(num, den);
    ls.setInitialOutputDerivatives(ydy0);
    ls.setInitialTime(0);
}

void printProgress(int width, float progress)
{
    std::cout << "[";
    int pos = width * progress;
    for (int i = 0; i < width; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
    if (progress == 1.0)
        std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_updates_takes_too_long)
{
    std::cout << "[TEST] update timeout" << std::endl;
    LinearSystem sys;
    sys.useNFilters(2);
    sys.setSampling(0.1);
    sys.setMaximumTimeBetweenUpdates(1);
    sys.setIntegrationMethod(IntegrationMethod::BACKWARD_EULER);

    Poly num(3), den(3);
    num << 0, 0, 1;
    den << 1, 2, 1;
    sys.setFilter(num, den);

    Eigen::MatrixXd ydy0(2,2);
    ydy0 <<   0, 0,
            0.5, 0;
    sys.setInitialOutputDerivatives(ydy0);
    sys.discretizeSystem();

    Eigen::MatrixXd u0 = Eigen::MatrixXd::Zero(2,2);
    sys.setInitialState(u0);
    sys.setInitialTime(0);

    std::cout << "[INFO] not to worry, the following warning is expected" << std::endl;
    Eigen::RowVectorXd y1, y2;
    Eigen::VectorXd u(2);
    u << 1, 1.5;
    sys.update(u, LinearSystem::getTimeFromSeconds(0.5));
    y1 = sys.update(u, LinearSystem::getTimeFromSeconds(1));
    y2 = sys.update(u, LinearSystem::getTimeFromSeconds(2.1));

    if ( (y1 - y2).cwiseAbs().maxCoeff() > std::numeric_limits<double>::min() )
    {
        BOOST_ERROR("outputs should be the same if the update took too long to be processed");
    }

    y1 = sys.update(u, LinearSystem::getTimeFromSeconds(3));
    // THESE LIMITS HAVE BEEN HARD CODED!
    if ( (y1 - y2).cwiseAbs().minCoeff() < 0.139545 && (y1 - y2).cwiseAbs().maxCoeff() > 0.142949 )
    {
        BOOST_ERROR("the filter values do not look right after calling update");
    }
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_number_of_filters_simple)
{
    std::cout << "[TEST] number of filters (simple test)" << std::endl;
    LinearSystem sys;
    sys.useNFilters(3);
    Poly num(2), den(2);
    num << 0, 1;
    den << 1, 1;
    BOOST_TEST_PASSPOINT();
    sys.setFilter(num, den);

    Eigen::MatrixXd ydy0(3,1);
    ydy0 << 1,
            1,
            1;
    BOOST_TEST_PASSPOINT();
    sys.setInitialOutputDerivatives(ydy0);

    BOOST_TEST_PASSPOINT();
    sys.discretizeSystem();

    Eigen::VectorXd u0 = Eigen::VectorXd::Zero(3,1);
    BOOST_TEST_PASSPOINT();
    sys.setInitialState(u0);

    BOOST_TEST_PASSPOINT();
    Eigen::VectorXd input(3);
    input << 2,2,2;
    sys.setInitialTime(0);
    Eigen::VectorXd out = sys.update(input, LinearSystem::getTimeFromSeconds(sys.getSampling()));
    double delta = 1e-15;
    if (std::abs(out(0) - out(1)) > delta || std::abs(out(1) != out(2)) > delta)
    {
        BOOST_ERROR("filters output differ");
    }
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_number_of_filters)
{
    std::cout << "[TEST] number of filters" << std::endl;
    BOOST_TEST_PASSPOINT();
    YAML::Node doc = YAML::LoadFile("test_LinearSystem.yml");

    int progress_width = 50;
    float progress_max = doc.size();
    double Ts;
    for (unsigned i = 0; i < doc.size(); i++)
    {
        printProgress(progress_width, i / progress_max);

        int n, order;
        BOOST_TEST_PASSPOINT();
        LinearSystem tustin_ls;
        BOOST_TEST_PASSPOINT();
        LinearSystem fwd_ls;
        BOOST_TEST_PASSPOINT();
        LinearSystem bwd_ls;
        BOOST_TEST_PASSPOINT();
        Eigen::VectorXd u;
        Eigen::VectorXd y_tustin_data;
        Eigen::VectorXd y_fwd_data;
        Eigen::VectorXd y_bwd_data;

        BOOST_TEST_PASSPOINT();
        GetYmlData(doc[i],n,order,tustin_ls,u,y_tustin_data,y_fwd_data,y_bwd_data,Ts);

        // Should change this to consider the exact same initial conditions as the
        // used to generate the test results
        Eigen::MatrixXd u0 = Eigen::MatrixXd::Constant(1, order, u(0));
        //
        Eigen::VectorXd u_i(1);
        Eigen::VectorXd y_tustin(n);
        //
        Eigen::VectorXd num, den;

        // Tustin
        BOOST_TEST_PASSPOINT();
        tustin_ls.setIntegrationMethod(IntegrationMethod::TUSTIN);
        tustin_ls.discretizeSystem();
        tustin_ls.setInitialState(u0);

        // update filters
        for (int k = 0; k < n; ++k)
        {
            Time time = LinearSystem::getTimeFromSeconds( (k+1) * Ts );
            u_i(0) = u(k);

            BOOST_TEST_PASSPOINT();
            y_tustin(k) = tustin_ls.update(u_i, time)(0);
        }

        double delta = 1e-5;
        double maxError;

        maxError = (y_tustin_data - y_tustin).cwiseAbs().maxCoeff();
        if (maxError > delta)
        {
                BOOST_ERROR("y_tustin error");
                std::cout << "max error = " << maxError << std::endl;
        }
    }
    printProgress(progress_width, 1.0);
    std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_LinearSystem)
{
    std::cout << "[TEST] Tustin, Forward Euler and Backward Euler" << std::endl;
    BOOST_TEST_PASSPOINT();
    YAML::Node doc = YAML::LoadFile("test_LinearSystem.yml");

    int progress_width = 50;
    float progress_max = doc.size();
    double Ts;
    for (unsigned i = 0; i < doc.size(); i++)
    {
        printProgress(progress_width, i / progress_max);

        int n, order;
        BOOST_TEST_PASSPOINT();
        LinearSystem tustin_ls;
        BOOST_TEST_PASSPOINT();
        LinearSystem fwd_ls;
        BOOST_TEST_PASSPOINT();
        LinearSystem bwd_ls;
        BOOST_TEST_PASSPOINT();
        Eigen::VectorXd u;
        Eigen::VectorXd y_tustin_data;
        Eigen::VectorXd y_fwd_data;
        Eigen::VectorXd y_bwd_data;

        BOOST_TEST_PASSPOINT();
        GetYmlData(doc[i],n,order,tustin_ls,u,y_tustin_data,y_fwd_data,y_bwd_data,Ts);

        fwd_ls = tustin_ls;
        bwd_ls = tustin_ls;

        // Should change this to consider the exact same initial conditions as the
        // used to generate the test results
        Eigen::MatrixXd u0 = Eigen::MatrixXd::Constant(1, order, u(0));
        //
        Eigen::VectorXd u_i(1);
        Eigen::VectorXd y_tustin(n);
        Eigen::VectorXd y_fwd(n);
        Eigen::VectorXd y_bwd(n);
        //
        Poly num, den;

        // Tustin
        BOOST_TEST_PASSPOINT();
        tustin_ls.setIntegrationMethod(IntegrationMethod::TUSTIN);
        tustin_ls.discretizeSystem();
        tustin_ls.setInitialState(u0);

        // Forward Euler
        BOOST_TEST_PASSPOINT();
        fwd_ls.setIntegrationMethod(IntegrationMethod::FORWARD_EULER);
        fwd_ls.discretizeSystem();
        fwd_ls.setInitialState(u0);

        // Backward Euler
        BOOST_TEST_PASSPOINT();
        bwd_ls.setIntegrationMethod(IntegrationMethod::BACKWARD_EULER);
        bwd_ls.discretizeSystem();
        bwd_ls.setInitialState(u0);

        // update filters
        for (int i = 0; i < n; i++)
        {
            Time time = LinearSystem::getTimeFromSeconds( (i+1) * Ts );
            u_i(0) = u(i);

            BOOST_TEST_PASSPOINT();
            y_tustin(i) = tustin_ls.update(u_i, time)(0);

            BOOST_TEST_PASSPOINT();
            y_fwd(i) = fwd_ls.update(u_i, time)(0);

            BOOST_TEST_PASSPOINT();
            y_bwd(i) = bwd_ls.update(u_i, time)(0);
        }

        double delta = 1e-5;
        double maxError;

        maxError = (y_tustin_data - y_tustin).cwiseAbs().maxCoeff();
        if (maxError > delta)
        {
                BOOST_ERROR("y_tustin error");
                std::cout << "max error = " << maxError << std::endl;
        }
        maxError = (y_fwd_data - y_fwd).cwiseAbs().maxCoeff();
        if (maxError > delta)
        {
                BOOST_ERROR("y_fwd error");
                std::cout << "max error = " << maxError << std::endl;
        }
        maxError = (y_bwd_data - y_bwd).cwiseAbs().maxCoeff();
        if (maxError > delta)
        {
                BOOST_ERROR("y_bwd error");
                std::cout << "max error = " << maxError << std::endl;
        }
    }
    printProgress(progress_width, 1.0);
    std::cout << std::endl;
}
