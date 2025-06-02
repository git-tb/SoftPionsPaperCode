#ifndef FILEREADER_H
#define FILEREADER_H

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <regex>

struct csvdata
{
    std::vector<std::string> header;
    std::vector<std::vector<double>> data;
};
csvdata readcsv(std::string source, std::string delimiter = ",", bool has_header = true, bool transpose = true)
{
    /*
        reads a csv file in the following format

        \
            # comment
            # comment
            # comment
            title1,title2,title3
            0.1,5.1,2.3
            0.2,5.7,1.8
            0.3,5.5,1.9
            ...
            0.9,4.7,3.2
        \

        and returns an object of type csvdata where

            csvdata.header = {title1,title2,title3} (empty if has_header == false and no header line exists)
            csvdata.data = {{0.1,5.1,2.3},
                            {0.2,5.7,1.8},
                            {0.3,5.5,1.9},
                            {...},
                            {0.9,4.7,3.2}   }

        If transpose == true, the data vector is instead ordered as

            csvdata.data = {    {0.1,0.2,0.3,...,0.9},
                                {5.1,5.7,5.5,...,4.7},
                                {2.3,1.8,1.9,...,3.2}   } 
    */

    std::ifstream input(source);

    if (!input.is_open())
    {
        std::cerr << "Error opening the file: " << source << " to read in" << std::endl;
        exit(1);
        return {{"ERROR"}, {{-1}}};
    }

    csvdata mydata;

    std::string line;

    if (has_header)
    {
        do {
            std::getline(input,line);
        } while (line.starts_with('#'));
        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos)
        {
            mydata.header.push_back(line.substr(0, pos));
            line.erase(0, pos + delimiter.length());
        }
        mydata.header.push_back(line.substr(0, pos));
    }

    while (std::getline(input, line))
    {
        while (line.starts_with('#')) std::getline(input,line);

        mydata.data.push_back(std::vector<double>());

        size_t pos = 0;
        std::string token;
        while ((pos = line.find(delimiter)) != std::string::npos)
        {
            mydata.data.back().push_back(std::stof(line.substr(0, pos)));
            line.erase(0, pos + delimiter.length());
        }
        mydata.data.back().push_back(std::stof(line.substr(0, pos)));
    }
    input.close();

    if (transpose)
    {
        std::vector<std::vector<double>> transposeddata(mydata.data[0].size());
        for (int i = 0; i < mydata.data[0].size(); i++)
        {
            for (int j = 0; j < mydata.data.size(); j++)
            {
                transposeddata[i].push_back(mydata.data[j][i]);
            }
        }

        mydata.data = transposeddata;
    }

    return mydata;
}

#endif