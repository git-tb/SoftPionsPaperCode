#ifndef UNINITFUNC_H
#define UNINITFUNC_H

#include <functional>
#include <cassert>
#include "debugmsg.h"

std::function<double(double)> UNINITIALIZED_FUNCTION = [](double x){
    DEBUGMSG("THIS FUNCTION HAS NOT BEEN INITIALIZED");
    exit(-1);
    return -1e10;
};

#endif