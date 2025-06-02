#ifndef INITDATA_H
#define INITDATA_H

#include <functional>

#include <functional>       // std::function
#include <string>           // std::string
#include <cmath>            // M_PI
#include <complex>          // std::complex<double>
#include <boost/math/interpolators/pchip.hpp>   // pchip interpolation

#include "filereader.h"     // csvdata, readcsv()
#include "uninitfunc.h"     // UNINITIALIZED_FUNCTION
#include "extrapolate.h"    // linearExtrapolate()

struct initFunctions {
    std::function<double(double)>   f0Re = UNINITIALIZED_FUNCTION, 
                                    f0Im = UNINITIALIZED_FUNCTION,
                                    Df0Re = UNINITIALIZED_FUNCTION,
                                    Df0Im = UNINITIALIZED_FUNCTION;
};

struct initData {
    csvdata initCSV;
    initFunctions initFunc;
};

initData ProcessInitialData(std::string datafile = "NO_INITIALDATA_SPECIFIED")
{ 
    // READ DATA
    csvdata initCSV = readcsv(datafile, ",", true, true);

    // INTERPOLATE
    std::vector<double> x = initCSV.data[0];
    std::vector<double> x1(x), x2(x), x3(x), x4(x);
    std::vector<double> 
        y1(initCSV.data[1]),
        y2(initCSV.data[2]),
        y3(initCSV.data[3]),
        y4(initCSV.data[4]);

    linearExtrapolate(x1, y1, 0 - 0.01);
    linearExtrapolate(x1, y1, M_PI / 2.0 + 0.01);
    linearExtrapolate(x2, y2, 0 - 0.01);
    linearExtrapolate(x2, y2, M_PI / 2.0 + 0.01);
    linearExtrapolate(x3, y3, 0 - 0.01);
    linearExtrapolate(x3, y3, M_PI / 2.0 + 0.01);
    linearExtrapolate(x4, y4, 0 - 0.01);
    linearExtrapolate(x4, y4, M_PI / 2.0 + 0.01);

    using boost::math::interpolators::pchip;
    auto f0Re_spline = pchip(
        std::move(x1),
        std::move(y1));
    auto f0Im_spline = pchip(
        std::move(x2),
        std::move(y2));
    auto Df0Re_spline = pchip(
        std::move(x3),
        std::move(y3));
    auto Df0Im_spline = pchip(
        std::move(x4),
        std::move(y4));
            
    // pack the raw csv data and the interpolated curves in an object and return
    initData mydata;
    mydata.initCSV = initCSV;

    mydata.initFunc.f0Re = std::move(f0Re_spline);  // not sure if std::move is necessary
    mydata.initFunc.f0Im = std::move(f0Im_spline);
    mydata.initFunc.Df0Re = std::move(Df0Re_spline);
    mydata.initFunc.Df0Im = std::move(Df0Im_spline);

    return mydata;
}

#endif