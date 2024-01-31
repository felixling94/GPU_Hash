#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <../include/statisch_hashtabelle.h>
#include <../core/hashtabelle.cpp>
#include <../core/statisch_hashtabelle.cpp>

#include <../src/hashtabelle.cuh>
#include <../src/statisch_hashtabelle.cuh>
#include <../src/test_main.h>

template <typename T1, typename T2>
void fuehrehashverfahren(hashtyp pHashtyp, size_t pGroesseHashtabelle,
std::vector<T1> pSchluesselVektor, std::vector<T2> pWerteVektor){
  Time Startuhr, Endeuhr;

  Startuhr = beginn_ende();

  //Erstelle eine statische Hashtabelle
  Statisch_Hashtabelle<T1,T2> hashtabelle1(pHashtyp,pGroesseHashtabelle);
  
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  if (pHashtyp == linear_aufloesung){
    std::cout << "Lineare Hashverfahren" << std::endl;
  }else{
    std::cout << "Quadratische Hashverfahren" << std::endl;
  }
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  
  hashtabelle1.insert_List(pSchluesselVektor.data(),pWerteVektor.data(),pSchluesselVektor.size());

  hashtabelle1.drucken();

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden;
  Endeuhr = beginn_ende();
  Hashmillisekunden = get_dauer(Startuhr,Endeuhr);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei ";

  if (pHashtyp == linear_aufloesung){
    std::cout << "linearen Hashverfahren: ";
  }else{
    std::cout << "quadratischen Hashverfahren: ";
  }
  std::cout << Hashmillisekunden << " Millisekunden." << std::endl;
};

int main(int argc, char** argv){
  long long int usrGroesseSchluessel, usrGroesseHashtabelle, maxSchluessel;
  uint32_t usrMaxSchluessel;
  double auslastungsfaktor;
  Time Startuhr1, Endeuhr1;
  
  if(argc < 4){
    std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
    return -1;
  }

  maxSchluessel = atoll(argv[1]);
  usrGroesseSchluessel = atoll(argv[2]);
  auslastungsfaktor = atof(argv[3]);

  if (auslastungsfaktor == 0){
    std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
    return -1;
  }

  if (maxSchluessel<=0){
    std::cout << "Der maximale Schlüssel muss mehr als 0 sein" << std::endl;
    return -1;
  }

  usrMaxSchluessel = (uint32_t) maxSchluessel;
  usrGroesseHashtabelle = (long long int) ceil((double) usrGroesseSchluessel / auslastungsfaktor);

  std::vector<uint32_t> schluesselListe = erzeuge_schluessel_zelle<uint32_t>(usrMaxSchluessel, usrGroesseSchluessel);
  std::vector<uint32_t> werteListe = erzeuge_werte_zelle<uint32_t>(usrMaxSchluessel, usrGroesseSchluessel);
  
  Startuhr1 = beginn_ende();

  //Beginn des Tests
  std::cout << "Tests für Einfügung von " << usrGroesseSchluessel << " Schlüsseln in die Hashtabelle" << std::endl;
  
  /////////////////////////////////////////////////////////////////////////////////////////
  //Lineare Hashverfahren
  /////////////////////////////////////////////////////////////////////////////////////////
  fuehrehashverfahren<uint32_t,uint32_t>(linear_aufloesung,usrGroesseHashtabelle,schluesselListe,werteListe);

  /////////////////////////////////////////////////////////////////////////////////////////
  //Quadratische Hashverfahren
  /////////////////////////////////////////////////////////////////////////////////////////
  fuehrehashverfahren<uint32_t,uint32_t>(quadratisch_aufloesung,usrGroesseHashtabelle,schluesselListe,werteListe);

  //Fasse Resultate zusammen
  double millisekunden;
  Endeuhr1 = beginn_ende();
  millisekunden = get_dauer(Startuhr1,Endeuhr1);

  std::cout << "Gesamtdauer für alle offenen Hashverfahren: " << millisekunden << " Millisekunden." << std::endl;
  std::cout << "Erfolgreich" << std::endl;

  return 0;
};