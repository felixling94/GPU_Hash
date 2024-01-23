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

template <typename T0, typename T1, typename T2>
void fuehrehashverfahren(hashtyp pHashtyp, T0 pGroesseHashtabelle, std::vector<Knoten<T1,T2>> pSchluesselVektor){
  Time Startuhr, Endeuhr;
  Startuhr = beginn_ende();

  //Erstelle eine Hashtabelle
  Hashtabelle<T0,T1,T2> pHashtabelle(pHashtyp,pGroesseHashtabelle);

  //Erstelle eine Tabelle von Kollisionen
  Kollision<T0,T1,T2> * pKollision(new Kollision<T0,T1,T2>[pSchluesselVektor.size()]);

  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  if (pHashtyp == linear_aufloesung){
    std::cout << "Lineare Hashverfahren" << std::endl;
  }else{
    std::cout << "Quadratische Hashverfahren" << std::endl;
  }
  std::cout << "****************************************************************";
  std::cout << "****************************************************************" << std::endl;
  
  pHashtabelle.insert_CUDA(pSchluesselVektor.data(),pSchluesselVektor.size(),pKollision); 

  T0 summe_kollision = 0;
  for (T0 i=0; i<pSchluesselVektor.size();i++) summe_kollision+=pKollision[i].zahlKollision;
    
  if(summe_kollision>0){
    std::cout << "Es gibt genau " << summe_kollision << " Kollision(en) und " << (double)summe_kollision/(double)pSchluesselVektor.size();
    std::cout << " Kollisionen pro Datenelement bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }else{
    std::cout << "Es gibt keine Kollisionen bei der Einfügung von Datenelementen in die Hashtabelle." << std::endl;
  }

  //Fasse Resultate für jede Hashverfahren zusammen
  double Hashmillisekunden;
  Endeuhr = beginn_ende();
  Hashmillisekunden = get_dauer(Startuhr,Endeuhr);

  std::cout << std::endl;
  std::cout << "Gesamtdauer bei ";

  if (pHashtyp == linear_aufloesung){
    std::cout << "linearen Hashverfahren ";
  }else{
    std::cout << "quadratischen Hashverfahren ";
  }
  std::cout << ": " << Hashmillisekunden << " Millisekunden." << std::endl;
  std::cout << std::endl;
}

int main(int argc, char** argv){
  long long int usrGroesseSchluessel, usrGroesseHashtabelle;
  double auslastungsfaktor;

  Time Startuhr1, Endeuhr1;
  
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
  
  Startuhr1 = beginn_ende();

  //Beginn des Tests
  std::cout << "Tests für Einfügung von " << insert_schluessel.size();
  std::cout << " Datenelementen in die Hashtabelle" << std::endl;
  
  /////////////////////////////////////////////////////////////////////////////////////////
  //Lineare Hashverfahren
  /////////////////////////////////////////////////////////////////////////////////////////
  fuehrehashverfahren<uint32_t,uint32_t,uint32_t>(linear_aufloesung,usrGroesseHashtabelle,insert_schluessel);
  
  /////////////////////////////////////////////////////////////////////////////////////////
  //Quadratische Hashverfahren
  /////////////////////////////////////////////////////////////////////////////////////////
  fuehrehashverfahren<uint32_t,uint32_t,uint32_t>(quadratisch_aufloesung,usrGroesseHashtabelle,insert_schluessel);

  //Fasse Resultate zusammen
  double millisekunden;
  Endeuhr1 = beginn_ende();
  millisekunden = get_dauer(Startuhr1,Endeuhr1);

  std::cout << "Gesamtdauer für alle offenen Hashverfahren: " << millisekunden << " Millisekunden.";
  std::cout << "" << std::endl;
    
  std::cout << "Erfolgreich" << std::endl;

  return 0;
}