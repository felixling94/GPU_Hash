#ifndef TEST_MAIN_H
#define TEST_MAIN_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <../include/statisch_hashtabelle.h>

//Zeit messen
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time beginn_ende(){
  return std::chrono::high_resolution_clock::now();
};

double get_dauer(Time start,Time ende){
    std::chrono::duration<double> differenz = ende - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(differenz);
    return us.count() / 1000.0f;
};

//////////////////////////////////////////////////////////////////////////////////////
//Funktionen für statische Hashtabelle
//////////////////////////////////////////////////////////////////////////////////////
//Erzeuge verschiedene Schlüssel zufällig
template <typename T1>
std::vector<T1> erzeuge_schluessel_zelle(size_t maxSchluessel, size_t zahlschluessel){
  std::random_device zufallCPU;
  size_t seed = zufallCPU();
  std::mt19937 rnd(seed);
  
  std::uniform_int_distribution<T1> distribution(1, maxSchluessel);
  
  std::vector<T1> schluesselVector;
  schluesselVector.reserve(zahlschluessel);

  for (size_t i = 0; i < zahlschluessel; i++){
    T1 rand;
    rand = distribution(rnd);
    schluesselVector.push_back(rand);
  }

  return schluesselVector;
};

template <typename T2>
std::vector<T2> erzeuge_werte_zelle(size_t maxWerte, size_t zahlWerte){
  std::random_device zufallCPU;
  size_t seed = zufallCPU();
  std::mt19937 rnd(seed);
  
  std::uniform_int_distribution<T2> distribution(1, maxWerte);
  
  std::vector<T2> werteVector;
  werteVector.reserve(zahlWerte);

  for (size_t i = 0; i < zahlWerte; i++){
    T2 rand;
    rand = distribution(rnd);
    werteVector.push_back(rand);
  }

  return werteVector;
};

//Mische verschiedene Schlüssel
template <typename T1>
std::vector<T1> mischen_schluessel_zelle(std::vector<T1> schluesselVector){
  std::random_device zufallCPU;
  size_t seed = zufallCPU();
  std::mt19937 rnd(seed);
  
  std::shuffle(schluesselVector.begin(), schluesselVector.end(), rnd);
  std::vector<T1> gemischtSchluessel;
  gemischtSchluessel.resize(schluesselVector.size());
  std::copy(schluesselVector.begin(), schluesselVector.begin() + schluesselVector.size(), gemischtSchluessel.begin());
  return gemischtSchluessel;
};

//////////////////////////////////////////////////////////////////////////////////////
//Funktionen für dynamische Hashtabelle
//////////////////////////////////////////////////////////////////////////////////////
//TODO

#endif