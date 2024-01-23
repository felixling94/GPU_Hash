#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

#include <../include/hashtabelle.h>

//Zeit messen
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time beginn_ende(){
  return std::chrono::high_resolution_clock::now();
}

double get_dauer(Time start,Time ende){
    std::chrono::duration<double> differenz = ende - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(differenz);
    return us.count() / 1000.0f;
}

//Erzeuge verschiedene Schlüssel zufällig bei Knoten
template <typename T0, typename T1, typename T2>
std::vector<Knoten<T1,T2>> erzeuge_schluessel(std::mt19937& rnd, T0 maxSchluessel, T0 zahlschluessel){
    std::uniform_int_distribution<T1> distribution(0, maxSchluessel);

    std::vector<Knoten<T1,T2>> schluesselVector;
    schluesselVector.reserve(zahlschluessel);

    for (T0 i = 0; i < zahlschluessel; i++){
        T0 rand0, rand1;
        rand0 = distribution(rnd);
        rand1 = distribution(rnd);
        schluesselVector.push_back(Knoten<T1,T2>{rand0,rand1});
    }
    return schluesselVector;
}

//Mische verschiedene Schlüssel bei Knoten
template <typename T0, typename T1, typename T2>
std::vector<Knoten<T1,T2>> mischen_schluessel(std::mt19937& rnd, std::vector<Knoten<T1,T2>> schluesselVector, 
T0 zahlGemischtSchluessel){
    std::shuffle(schluesselVector.begin(), schluesselVector.end(), rnd);
    std::vector<Knoten<T1,T2>> gemischtSchluessel;
    gemischtSchluessel.resize(zahlGemischtSchluessel);
    std::copy(schluesselVector.begin(), schluesselVector.begin() + zahlGemischtSchluessel, gemischtSchluessel.begin());
    return gemischtSchluessel;
}

#endif
