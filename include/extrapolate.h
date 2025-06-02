#ifndef EXTRAPOLATE_H
#define EXTRAPOLATE_H

#include <iostream>
#include <vector>   // std::vector

void linearExtrapolate(std::vector<double> &x, std::vector<double> &y, double x_extr)
{
    std::vector<double> newx, newy;
    newx.push_back(x_extr);
    if (x_extr < x[0])
    {
        double slope = (y[1] - y[0]) / (x[1] - x[0]);
        newy.push_back(y[0] + slope * (x_extr - x[0]));
        x.insert(x.begin(), newx.begin(), newx.end());
        y.insert(y.begin(), newy.begin(), newy.end());
    }
    else if (x_extr > x[x.size()-1])
    {
        int n = x.size();
        double slope = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
        newy.push_back(y[n - 1] + slope * (x_extr - x[n - 1]));
        x.insert(x.end(), newx.begin(), newx.end());
        y.insert(y.end(), newy.begin(), newy.end());
    }
    else
    {
        std::cout << "WARNING: Extrapolation point must lie outside data range. No extrapolation performed." << std::endl;
    }
}

void addPoint(std::vector<double> &xx, std::vector<double> &yy, double x, double y)
{
    int i = 0;
    while(x > xx[i]) i++;
    xx.insert(std::next(xx.begin(),i),x);
    yy.insert(std::next(yy.begin(),i),y);
}

#endif