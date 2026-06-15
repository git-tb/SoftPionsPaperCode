/**
 * @file integratetest.cpp
 * @brief Test file for the integrate_nested function
 * g++ src/integratetest.cpp -I./include/ -I./Cuba-4.2.2 -L./Cuba-4.2.2 -lcuba -o integratetest
 * @version 0.1
 * @date 2026-06-15
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <iostream>
#include <functional>
#include <cmath>
#include "integrate_nested.h"

int main()  {

    std::function<double(double, double)> f = [](double x1, double x2) {
        return std::sin(x1*std::sin(x2));
    };

    std::function<double(double)> a2 = [](double x1) {
        return x1;
    };

    std::function<double(double)> b2 = [](double x1) {
        return std::pow(x1, 2);
    };

    double result, error, prob;
    integrate_nested(0.5, 1.5, a2, b2, f, result, error, prob, 1e-3, 1e-3, 1000);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Probability: " << prob << std::endl;

    return 0;
}