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
  uint32_t summe_kollision, summe_kollision1;
  
  Time Startuhr,Startuhr1, Endeuhr,Endeuhr1;

  summe_kollision = 0;
  summe_kollision1 = 0;
  
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

  //Beginn des Tests
  std::cout << "Tests für Einfügung von " << insert_schluessel.size();
  std::cout << " Datenelementen in die Hashtabelle" << std::endl;
  //////////////////////////////////////////////////////////////////////////////////
  //Cuckoo-Hashverfahren
  //////////////////////////////////////////////////////////////////////////////////
  Startuhr = beginn_ende();

  Hashtabelle<uint32_t,uint32_t,uint32_t> pHashtabelle1(cuckoo_aufloesung,usrGroesseHashtabelle);
  Hashtabelle<uint32_t,uint32_t,uint32_t> pHashtabelle2(cuckoo_aufloesung,usrGroesseHashtabelle); 

  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  std::cout << "Cuckoo-Hashverfahren" << std::endl;
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;

  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision1(new Kollision<uint32_t,uint32_t,uint32_t>[insert_schluessel.size()]);
 
  insert_Cuckoo_CUDA<uint32_t,uint32_t,uint32_t>(insert_schluessel.data(),insert_schluessel.size(),pHashtabelle1,pHashtabelle2,pKollision1);

  for (uint32_t i=0; i<insert_schluessel.size();i++) summe_kollision+=pKollision1[i].zahlKollision;

  if(summe_kollision>0){
    std::cout << "Es gibt genau " << summe_kollision << " Kollision(en) und " << (double)summe_kollision/(double)insert_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }
  
  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden;
  Endeuhr = beginn_ende();
  Hashmillisekunden = get_dauer(Startuhr,Endeuhr);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "Cuckoo-Hashverfahren" << ": " << Hashmillisekunden;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  std::vector<Knoten<uint32_t,uint32_t>> suchen_schluessel = mischen_schluessel<uint32_t,uint32_t,uint32_t>(rnd,insert_schluessel,insert_schluessel.size());

  //Beginn des Tests
  std::cout << "Tests für Suche nach " << suchen_schluessel.size();
  std::cout << " Datenelementen in der Hashtabelle" << std::endl;

  Startuhr1 = beginn_ende();

  //Erstelle eine Tabelle von Kollisionen
  Kollision<uint32_t,uint32_t,uint32_t> * pKollision2(new Kollision<uint32_t,uint32_t,uint32_t>[suchen_schluessel.size()]);

  suchen_Cuckoo_CUDA<uint32_t,uint32_t,uint32_t>(suchen_schluessel.data(),suchen_schluessel.size(),pHashtabelle1,pHashtabelle2,pKollision2);

  for (uint32_t i=0; i<suchen_schluessel.size();i++) summe_kollision1+=pKollision2[i].zahlKollision;
  
  if(summe_kollision1>0){
    std::cout << "Es gibt genau " << summe_kollision1 << " Kollision(en) und " << (double)summe_kollision1/(double)suchen_schluessel.size();
    std::cout << " Kollisionen pro Datenelement bei der Suche nach Datenelementen in der Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Suche nach Datenelementen in der Hashtabelle." << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden1;
  Endeuhr1 = beginn_ende();
  Hashmillisekunden1 = get_dauer(Startuhr1,Endeuhr1);

  std::cout << "" << std::endl;
  std::cout << "Gesamtdauer bei " << "Cuckoo-Hashverfahren" << ": " << Hashmillisekunden1;
  std::cout << " Millisekunden." << std::endl;
  std::cout << "" << std::endl;

  std::cout << "Erfolgreich" << std::endl;

  return 0;
}