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
  Time Startuhr1, Endeuhr1;
  
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
  
  Startuhr1 = beginn_ende();

  //Beginn des Tests
  std::cout << "Tests für Einfügung von " << insert_schluessel.size();
  std::cout << " Datenelementen in die Hashtabelle" << std::endl;
  //////////////////////////////////////////////////////////////////////////////////
  //Lineare Hashverfahren
  //////////////////////////////////////////////////////////////////////////////////
  Time Startuhr2, Endeuhr2;
  Startuhr2 = beginn_ende();

  Hashtabelle<uint32_t,uint32_t,uint32_t> pHashtabelle1(linear_aufloesung,usrGroesseHashtabelle);

  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  std::cout << "Lineare Hashverfahren" << std::endl;
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  
  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision1(new Kollision<uint32_t,uint32_t,uint32_t>[insert_schluessel.size()]);

  pHashtabelle1.insert_CUDA(insert_schluessel.data(),insert_schluessel.size(),pKollision1); 
  
  if (boolStanddruecke == true){
    pHashtabelle1.schalteStandDruecke();
    pHashtabelle1.druckeHashtabelle();

    uint32_t summe_kollision = 0;

    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << "  ";
    std::cout << "Zahl der Kollisionen" << std::endl;
    for (uint32_t i = 0; i<insert_schluessel.size(); i++){
      std::cout << i << "  " << pKollision1[i].knoten.schluessel << "  ";
      std::cout << pKollision1[i].knoten.wert << "  " << pKollision1[i].zahlKollision  << std::endl;
      summe_kollision+=pKollision1[i].zahlKollision;
    }

    std::cout << "Gesamtsumme der Kollisionen: " << summe_kollision << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden2;
  Endeuhr2 = beginn_ende();
  Hashmillisekunden2 = get_dauer(Startuhr2,Endeuhr2);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "linearen Hashverfahren" << ": " << Hashmillisekunden2;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  Time Startuhr3, Endeuhr3;
  
  std::vector<Knoten<uint32_t,uint32_t>> suchen_schluessel = mischen_schluessel<uint32_t,uint32_t,uint32_t>(rnd,insert_schluessel,insert_schluessel.size());
  Startuhr3 = beginn_ende();

  //Beginn des Tests
  std::cout << "Tests für Suche nach " << suchen_schluessel.size();
  std::cout << " Datenelementen in der Hashtabelle" << std::endl;

  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision2(new Kollision<uint32_t,uint32_t,uint32_t>[suchen_schluessel.size()]);
  
  pHashtabelle1.suchen_CUDA(suchen_schluessel.data(),suchen_schluessel.size(),pKollision2);

  if (boolStanddruecke == true){
    uint32_t summe_kollision = 0;

    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << "  ";
    std::cout << "Zahl der Kollisionen" << std::endl;
    for (uint32_t i = 0; i<suchen_schluessel.size(); i++){
      std::cout << i << "  " << pKollision2[i].knoten.schluessel << "  ";
      std::cout << pKollision2[i].knoten.wert << "  " << pKollision2[i].zahlKollision  << std::endl;
      summe_kollision+=pKollision2[i].zahlKollision;
    }

    std::cout << "Gesamtsumme der Kollisionen: " << summe_kollision << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden3;
  Endeuhr3 = beginn_ende();
  Hashmillisekunden3 = get_dauer(Startuhr3,Endeuhr3);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "linearen Hashverfahren" << ": " << Hashmillisekunden3;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  //////////////////////////////////////////////////////////////////////////////////
  //Quadratische Hashverfahren
  //////////////////////////////////////////////////////////////////////////////////
  Time Startuhr4, Endeuhr4;
  Startuhr4 = beginn_ende();
  
  Hashtabelle<uint32_t,uint32_t,uint32_t> pHashtabelle2(quadratisch_aufloesung,usrGroesseHashtabelle);

  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  std::cout << "Quadratische Hashverfahren" << std::endl;
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  
  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision3(new Kollision<uint32_t,uint32_t,uint32_t>[insert_schluessel.size()]);

  pHashtabelle2.insert_CUDA(insert_schluessel.data(),insert_schluessel.size(),pKollision3); 
  
  if (boolStanddruecke == true){
    pHashtabelle2.schalteStandDruecke();
    pHashtabelle2.druckeHashtabelle();

    uint32_t summe_kollision = 0;

    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << "  ";
    std::cout << "Zahl der Kollisionen" << std::endl;
    for (uint32_t i = 0; i<insert_schluessel.size(); i++){
      std::cout << i << "  " << pKollision3[i].knoten.schluessel << "  ";
      std::cout << pKollision3[i].knoten.wert << "  " << pKollision3[i].zahlKollision  << std::endl;
      summe_kollision+=pKollision3[i].zahlKollision;
    }

    std::cout << "Gesamtsumme der Kollisionen: " << summe_kollision << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden4;
  Endeuhr4 = beginn_ende();
  Hashmillisekunden4 = get_dauer(Startuhr4,Endeuhr4);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "quadratischen Hashverfahren" << ": " << Hashmillisekunden4;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  Time Startuhr5, Endeuhr5;
  
  Startuhr5 = beginn_ende();

  //Beginn des Tests
  std::cout << "Tests für Suche nach " << suchen_schluessel.size();
  std::cout << " Datenelementen in der Hashtabelle" << std::endl;

  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision4(new Kollision<uint32_t,uint32_t,uint32_t>[suchen_schluessel.size()]);
  
  pHashtabelle2.suchen_CUDA(suchen_schluessel.data(),suchen_schluessel.size(),pKollision4);

  if (boolStanddruecke == true){
    uint32_t summe_kollision = 0;

    std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << "  ";
    std::cout << "Zahl der Kollisionen" << std::endl;
    for (uint32_t i = 0; i<suchen_schluessel.size(); i++){
      std::cout << i << "  " << pKollision4[i].knoten.schluessel << "  ";
      std::cout << pKollision4[i].knoten.wert << "  " << pKollision4[i].zahlKollision  << std::endl;
      summe_kollision+=pKollision4[i].zahlKollision;
    }

    std::cout << "Gesamtsumme der Kollisionen: " << summe_kollision << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden5;
  Endeuhr5 = beginn_ende();
  Hashmillisekunden5 = get_dauer(Startuhr5,Endeuhr5);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "quadratischen Hashverfahren" << ": " << Hashmillisekunden5;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;
  
  //Fasse Resultate zusammen
  double millisekunden1;
  Endeuhr1 = beginn_ende();
  millisekunden1 = get_dauer(Startuhr1,Endeuhr1);

  std::cout << "Gesamtdauer für alle offenen Hashverfahren: " << millisekunden1 << " Millisekunden.";
  std::cout << "" << std::endl;
    
  std::cout << "Erfolgreich" << std::endl;

  return 0;
}