#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <../include/hashtabelle.h>
#include <../include/test_main.h>
#include <../core/hashtabelle.cpp>
#include <../src/hashtabelle_kernels_insert.cuh>
#include <../src/hashtabelle_kernels_suchen.cuh>

int main(int argc, char** argv){
  long long int usrGroesseSchluessel, usrGroesseHashtabelle, maxSchluessel;
  int standdruecke;
  uint32_t usrMaxSchluessel;
  double auslastungsfaktor;
  bool boolStanddruecke;
  Time Startuhr, Endeuhr;
  
  if(argc < 5){
    std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
    return -1;
  }

  maxSchluessel = atoll(argv[1]);
  usrGroesseSchluessel = atoll(argv[2]);
  auslastungsfaktor = atof(argv[3]);
  standdruecke = atoi(argv[4]);

  if (auslastungsfaktor == 0){
    std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
    return -1;
  }

  if (maxSchluessel<=0){
    std::cout << "Der maximale Schlüssel muss mehr als 0 sein" << std::endl;
    return -1;
  }

  if (standdruecke<0 || standdruecke>1){
    std::cout << "Der Stand des Druckens muss entweder 0 oder 1 sein." << std::endl;
    return -1;
  }

  usrMaxSchluessel = (uint32_t) maxSchluessel;
  usrGroesseHashtabelle = (long long int) ceil((double) usrGroesseSchluessel / auslastungsfaktor);

  if (standdruecke==1){
    boolStanddruecke = true;
  }else{
    boolStanddruecke = false;
  }

  std::random_device zufallsgeraet;
  uint32_t seed = zufallsgeraet();
  std::mt19937 rnd(seed);

  std::cout << "Zufallsgenerator: " << seed << std::endl;

  std::vector<Knoten<uint32_t,uint32_t>> insert_schluessel = erzeuge_schluessel<uint32_t,uint32_t,uint32_t>(rnd,usrMaxSchluessel, usrGroesseSchluessel);
  
  Startuhr = beginn_ende();

  //Beginn des Tests
  std::cout << "Tests für Einfügung von " << insert_schluessel.size();
  std::cout << " Datenelementen in die Hashtabelle" << std::endl;
  //////////////////////////////////////////////////////////////////////////////////
  //Cuckoo Hashverfahren
  //////////////////////////////////////////////////////////////////////////////////

  Hashtabelle<uint32_t,uint32_t,uint32_t> pHashtabelle1(cuckoo_aufloesung,usrGroesseHashtabelle);
  Hashtabelle<uint32_t,uint32_t,uint32_t> pHashtabelle2(cuckoo_aufloesung,usrGroesseHashtabelle);

  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  std::cout << "Cuckoo-Hashverfahren" << std::endl;
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;

  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision(new Kollision<uint32_t,uint32_t,uint32_t>[insert_schluessel.size()]);

  insert_Cuckoo_CUDA<uint32_t,uint32_t,uint32_t>(insert_schluessel.data(),insert_schluessel.size(),pHashtabelle1,pHashtabelle2, pKollision);

  uint32_t summe_kollision = 0;

  if (boolStanddruecke == true){
    std::cout << "****************************************************************";
    std::cout << "****************************************************************" << std::endl;
    std::cout << "1. Hashtabelle" << std::endl;
    std::cout << "****************************************************************";
    std::cout << "****************************************************************" << std::endl;
    pHashtabelle1.schalteStandDruecke();
    pHashtabelle1.druckeHashtabelle();

    std::cout << "****************************************************************";
    std::cout << "****************************************************************" << std::endl;
    std::cout << "2. Hashtabelle" << std::endl;
    std::cout << "****************************************************************";
    std::cout << "****************************************************************" << std::endl;
    pHashtabelle2.schalteStandDruecke();
    pHashtabelle2.druckeHashtabelle();
    
    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << "  ";
    std::cout << "Zahl der Kollisionen" << std::endl;
    for (uint32_t i = 0; i<insert_schluessel.size(); i++){
      std::cout << i << "     " << pKollision[i].knoten.schluessel << "     ";
      std::cout << pKollision[i].knoten.wert << "     " << pKollision[i].zahlKollision  << std::endl;
      summe_kollision+=pKollision[i].zahlKollision;
    }
  }else{
    for (uint32_t i = 0; i<insert_schluessel.size(); i++) summe_kollision+=pKollision[i].zahlKollision;
  }
    
  if(summe_kollision>0){
    std::cout << "Es gibt genau " << summe_kollision << " Kollision(en) und " << (double)summe_kollision/(double)insert_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }

  //Fasse Resultate für Cuckoo-Hashverfahren zusammen
  double Hashmillisekunden;
  Endeuhr = beginn_ende();
  Hashmillisekunden = get_dauer(Startuhr,Endeuhr);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "Cuckoo-Hashverfahren" << ": " << Hashmillisekunden;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  std::cout << "Erfolgreich" << std::endl;

  return 0;
}