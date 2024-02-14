#ifndef TIMER_H
#define TIMER_H

#include <iostream>
#include <ctime>
#include <chrono>

using zeitpunkt = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace Zeit{
    zeitpunkt grundStart;
    zeitpunkt grundEnde;    

    void grundStarte(){
        grundStart = std::chrono::high_resolution_clock::now();
    };
    
    void grundBeende(){
        grundEnde = std::chrono::high_resolution_clock::now();
    };

    double getGrundDauer(){
        std::chrono::duration<double> differenz = grundEnde - grundStart;
        std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(differenz);
        
        return us.count() / 1000.0f;
    };

    zeitpunkt start;
    zeitpunkt ende;

    void starte(){
        start = std::chrono::high_resolution_clock::now();
    };

    void beende(){
        ende = std::chrono::high_resolution_clock::now();
    };

    double getDauer(){
        std::chrono::duration<double> differenz = ende - start;
        std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(differenz);
        
        return us.count() / 1000.0f;
    };
};

#endif