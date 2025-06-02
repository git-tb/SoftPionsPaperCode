#ifndef FREEZEOUT_H
#define FREEZEOUT_H

#include <functional>       // std::function
#include <string>           // std::string
#include <cmath>            // M_PI
#include <complex>          // std::complex<double>
#include <boost/math/interpolators/pchip.hpp>   // pchip interpolation

#include "filereader.h"     // csvdata, readcsv()
#include "uninitfunc.h"     // UNINITIALIZED_FUNCTION
#include "extrapolate.h"    // linearExtrapolate()

struct freezeoutFunctions {
    std::function<double(double)>   tau = UNINITIALIZED_FUNCTION, 
                                    r = UNINITIALIZED_FUNCTION,
                                    Dtau = UNINITIALIZED_FUNCTION,
                                    Dr = UNINITIALIZED_FUNCTION;
};

struct freezeoutData {
    csvdata foCSV;
    freezeoutFunctions foFunc;
};

freezeoutData ProcessFreezeoutData(std::string datafile = "NO_FREEZEOUTDATA_SPECIFIED")
{
    // READ IN DATA
    csvdata foCSV = readcsv(datafile, ",", true, true);

    // INTERPOLATE
    std::vector<double> x = foCSV.data[0];
    std::vector<double> x1(x), x2(x), x3(x), x4(x);
    std::vector<double> 
        y1(foCSV.data[1]),
        y2(foCSV.data[2]),
        y3(foCSV.data[3]),
        y4(foCSV.data[4]);

    // tau
    linearExtrapolate(x1, y1, 0 - 0.01);
    linearExtrapolate(x1, y1, M_PI / 2.0 + 0.01);
    // r
    // addPoint(x2,y2,0,0);
    linearExtrapolate(x2, y2, 0 - 0.01);
    linearExtrapolate(x2, y2, M_PI / 2.0 + 0.01);
    // Dtau
    // addPoint(x3,y3,0,0);
    linearExtrapolate(x3, y3, 0 - 0.01);
    linearExtrapolate(x3, y3, M_PI / 2.0 + 0.01);
    // Dr
    linearExtrapolate(x4, y4, 0 - 0.01);
    linearExtrapolate(x4, y4, M_PI / 2.0 + 0.01);

    using boost::math::interpolators::pchip;
    auto tau_spline = pchip(
        std::move(x1),
        std::move(y1));
    auto r_spline = pchip(
        std::move(x2),
        std::move(y2));
    auto Dtau_spline = pchip(
        std::move(x3),
        std::move(y3));
    auto Dr_spline = pchip(
        std::move(x4),
        std::move(y4));

    // pack the raw csv data and the interpolated curves in an object and return
    freezeoutData mydata;
    mydata.foCSV = foCSV;

    mydata.foFunc.tau = std::move(tau_spline);      // std::move doesn't seem to be necessary, 
                                                    //  but it in principle it makes sense because we don't
                                                    //  want the splines to be destroyed at the end of the scope
    mydata.foFunc.r = std::move(r_spline);
    mydata.foFunc.Dtau = std::move(Dtau_spline);
    mydata.foFunc.Dr = std::move(Dr_spline);

    return mydata;
}

#endif