#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <../include/dynamisch_hashtabelle.h>
#include <../core/hashtabelle.cpp>
#include <../core/dynamisch_hashtabelle.cpp>

#include <../src/hashtabelle.cuh>
#include <../src/dynamisch_hashtabelle.cuh>
#include <../src/test_main.h>

template <typename T1, typename T2>
void fuehrehashverfahren(bool pStandAufloesung, size_t pGroesseHashtabelle,
std::vector<T1> pSchluesselVektor, std::vector<T2> pWerteVektor){
  Time Startuhr, Endeuhr;

  Startuhr = beginn_ende();

  //Erstelle eine dynamische Hashtabelle
  Dynamisch_Hashtabelle<T1,T2> hashtabelle(pStandAufloesung,pGroesseHashtabelle);
  
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  if (pStandAufloesung == true){
    std::cout << "Dynamische Hashverfahren" << std::endl;
  }else{
    std::cout << "Ohne Dynamische Hashverfahren" << std::endl;
  }
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  
  hashtabelle.insert_List(pSchluesselVektor.data(),pWerteVektor.data(),pSchluesselVektor.size());

  hashtabelle.drucken();

  //Fasse Resultate für die dynamischen Hashverfahren zusammen
  double Hashmillisekunden;
  Endeuhr = beginn_ende();
  Hashmillisekunden = get_dauer(Startuhr,Endeuhr);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei ";

  if (pStandAufloesung == true){
    std::cout << "dynamischen Hashverfahren: ";
  }else{
    std::cout << "Hashverfahren ohne Kollionsauflösung: ";
  }
  std::cout << Hashmillisekunden << " Millisekunden." << std::endl;
  std::cout << "Erfolgreich" << std::endl;
};

int main(int argc, char** argv){
  long long int usrGroesseSchluessel, usrGroesseHashtabelle, maxSchluessel;
  int standAufloesung;
  uint32_t usrMaxSchluessel;
  double auslastungsfaktor;
  bool boolStandAufloesung;

  if(argc < 5){
    std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
    return -1;
  }

  maxSchluessel = atoll(argv[1]);
  usrGroesseSchluessel = atoll(argv[2]);
  auslastungsfaktor = atof(argv[3]);
  standAufloesung = atoi(argv[4]);

  if (auslastungsfaktor == 0){
    std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
    return -1;
  }

  if (maxSchluessel<=0){
    std::cout << "Der maximale Schlüssel muss mehr als 0 sein" << std::endl;
    return -1;
  }
  
  if (standAufloesung<0 || standAufloesung>1){
    std::cout << "Der Stand der Kollionsauflösung muss entweder 0 oder 1 sein." << std::endl;
    return -1;
  }
  
  if (standAufloesung==1){
    boolStandAufloesung = true;
  }else{
    boolStandAufloesung = false;
  }

  usrMaxSchluessel = (uint32_t) maxSchluessel;
  usrGroesseHashtabelle = (long long int) ceil((double) usrGroesseSchluessel / auslastungsfaktor);

  std::vector<uint32_t> schluesselListe = erzeuge_schluessel_zelle<uint32_t>(usrMaxSchluessel, usrGroesseSchluessel);
  std::vector<uint32_t> werteListe = erzeuge_werte_zelle<uint32_t>(usrMaxSchluessel, usrGroesseSchluessel);

  //Beginn des Tests
  std::cout << "Tests für Einfügung von " << usrGroesseSchluessel << " Schlüsseln in die Hashtabelle" << std::endl;

  /////////////////////////////////////////////////////////////////////////////////////////
  //Dynamische Hashverfahren
  /////////////////////////////////////////////////////////////////////////////////////////
  fuehrehashverfahren<uint32_t,uint32_t>(boolStandAufloesung,usrGroesseHashtabelle,schluesselListe,werteListe);

  return 0;
};