#ifndef DEBUGMSG_H
#define DEBUGMSG_H

#include <iostream>

#define DEBUGMSG(str) do { std::cout << str << std::endl; } while( false )

#endif