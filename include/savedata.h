#ifndef SAVEDATA_H
#define SAVEDATA_H

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <string>
#include <functional>
#include <type_traits>

template <typename T>
concept Real = std::is_floating_point_v<T>;

template <typename T>
concept Complex = 
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

template <typename T>
requires (Real<T> || Complex<T>)
void writeFunctionsToFile(std::string path, 
    std::vector<std::function<T(double)>> funcs, 
    double a, 
    double b, 
    int Nsamples, 
    std::vector<std::string> headers, 
    std::vector<std::string> comments = std::vector<std::string>({}))
{
    assert(funcs.size() == headers.size()-1);

    std::ofstream output(path);

    if (!output.is_open()) {
        std::cerr << "Error opening the file: " << path << " to save to" << std::endl;
        return;
    }

    for(int i = 0; i < comments.size(); i++)
        output << "# " << comments[i] << std::endl;

    for(int i = 0; i < headers.size(); i++) {
        if (i == 0)
            output << headers[i];
        else if constexpr (Real<T>)
            output << headers[i];
        else if constexpr (Complex<T>)
            output << "Re@" << headers[i] <<  ",Im@" << headers[i];
        else    
            std::cerr << "Cannot convert function return type!" << std::endl;        
        if(i != headers.size()-1) output << ",";
    }
    output << std::endl;

    double dx = (b - a) / (Nsamples - 1);
    for (int i = 0; i < Nsamples; i++)
    {
        output << i * dx << ",";
        for(int j = 0; j < funcs.size(); j++)
        {
            std::function<T(double)> func = funcs[j];
            T y = func(i * dx);
            if constexpr (Real<T>)
                output << y;
            else if constexpr (Complex<T>)
                output << std::real(y) << "," << std::imag(y);
            else    
                std::cerr << "Cannot convert function return type!" << std::endl;
            if(j != funcs.size()-1) output << ",";
        }
        output << std::endl;
    }
    output.close();
}

#endif