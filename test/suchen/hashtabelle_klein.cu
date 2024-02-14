#include <stdint.h>
#include <iostream>
#include <string>

#include <../include/hashfunktionen.h>
#include <../include/test_parallel.cuh>
#include <../tools/timer.h>

int main(int argc, char** argv){
    long long int usrGroesseSchluessel, usrGroesseHashtabelle, maxSchluessel;
    uint32_t usrMaxSchluessel;
    double auslastungsfaktor;
  
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
    
    Test_Parallel<uint32_t,uint32_t> Test_Parallel(usrMaxSchluessel,usrMaxSchluessel,usrGroesseSchluessel,usrGroesseHashtabelle);

    Test_Parallel.erzeuge_schluessel();
    Test_Parallel.erzeuge_werte();

    Zeit::grundStarte();
  
    /////////////////////////////////////////////////////////////////////////////////////////
    //Keine Kollisonsauflösung
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Parallel.suche_hashtabelle(keine_aufloesung,murmer,true);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Lineare Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Parallel.suche_hashtabelle(linear_aufloesung,murmer,true);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Quadratische Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Parallel.suche_hashtabelle(quadratisch_aufloesung,murmer,true);
    
    //Fasse Resultate zusammen
    Zeit::grundBeende();
    std::cout << "Gesamtdauer für alle offenen Hashverfahren (in Millisekunden)     : ";
    std::cout << Zeit::getGrundDauer() << std::endl;
    std::cout << std::endl;
    std::cout << "Erfolgreich" << std::endl;

    return 0;
};