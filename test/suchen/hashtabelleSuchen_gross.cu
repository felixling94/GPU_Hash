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
  long long int usrGroesseSchluessel, usrGroesseHashtabelle;
  double auslastungsfaktor;

  Time Startuhr, Endeuhr;
  
  if(argc < 3){
    std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
    return -1;
  }

  usrGroesseSchluessel = atoll(argv[1]);
  auslastungsfaktor = atof(argv[2]);

  if (auslastungsfaktor == 0){
    std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
    return -1;
  }
  
  const uint32_t maxSchluessel = 100000000;
  usrGroesseHashtabelle = (long long int) ceil((double) usrGroesseSchluessel / auslastungsfaktor);
  
  std::random_device zufallsgeraet;
  uint32_t seed = zufallsgeraet();
  std::mt19937 rnd(seed);

  std::cout << "Zufallsgenerator: " << seed << std::endl;

  std::vector<Knoten<uint32_t,uint32_t>> insert_schluessel = erzeuge_schluessel<uint32_t,uint32_t,uint32_t>(rnd,maxSchluessel, usrGroesseSchluessel);
  
  Startuhr = beginn_ende();

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

  uint32_t summe_kollision1 = 0;
  for (uint32_t i=0; i<insert_schluessel.size();i++) summe_kollision1+=pKollision1[i].zahlKollision;
    
  if(summe_kollision1>0){
    std::cout << "Es gibt genau " << summe_kollision1 << " Kollision(en) und " << summe_kollision1/insert_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden1;
  Endeuhr2 = beginn_ende();
  Hashmillisekunden1 = get_dauer(Startuhr2,Endeuhr2);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "linearen Hashverfahren" << ": " << Hashmillisekunden1;
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

  uint32_t summe_kollision2 = 0;
  for (uint32_t i=0; i<insert_schluessel.size();i++) summe_kollision2+=pKollision2[i].zahlKollision;
    
  if(summe_kollision2>0){
    std::cout << "Es gibt genau " << summe_kollision2 << " Kollision(en) und " << summe_kollision2/insert_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden2;
  Endeuhr3 = beginn_ende();
  Hashmillisekunden2 = get_dauer(Startuhr3,Endeuhr3);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "linearen Hashverfahren" << ": " << Hashmillisekunden2;
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

  pHashtabelle2.insert_CUDA(insert_schluessel.data(),insert_schluessel.size(),pKollision2); 
  
  uint32_t summe_kollision3 = 0;
  for (uint32_t i=0; i<insert_schluessel.size();i++) summe_kollision3+=pKollision3[i].zahlKollision;
    
  if(summe_kollision3>0){
    std::cout << "Es gibt genau " << summe_kollision3 << " Kollision(en) und " << summe_kollision3/insert_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden3;
  Endeuhr4 = beginn_ende();
  Hashmillisekunden3 = get_dauer(Startuhr4,Endeuhr4);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "quadratischen Hashverfahren" << ": " << Hashmillisekunden3;
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
  
  uint32_t summe_kollision4 = 0;
  for (uint32_t i=0; i<insert_schluessel.size();i++) summe_kollision4+=pKollision4[i].zahlKollision;
    
  if(summe_kollision4>0){
    std::cout << "Es gibt genau " << summe_kollision4 << " Kollision(en) und " << summe_kollision4/insert_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden4;
  Endeuhr5 = beginn_ende();
  Hashmillisekunden4 = get_dauer(Startuhr5,Endeuhr5);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "linearen Hashverfahren" << ": " << Hashmillisekunden4;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  //Fasse Resultate zusammen
  double millisekunden;
  Endeuhr = beginn_ende();
  millisekunden = get_dauer(Startuhr,Endeuhr);

  std::cout << "Gesamtdauer für alle offenen Hashverfahren: " << millisekunden << " Millisekunden.";
  std::cout << "" << std::endl;
    
  std::cout << "Erfolgreich" << std::endl;

  return 0;
}