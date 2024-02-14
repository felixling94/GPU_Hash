#include <stdint.h>
#include <iostream>
#include <string>

#include <../include/hashfunktionen.h>
#include <../include/test_parallel.cuh>
#include <../tools/timer.h>

int main(int argc, char** argv){
    long long int usrGroesseSchluessel, usrGroesseHashtabelle;
    double auslastungsfaktor;
    
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

    Test_Parallel<uint32_t,uint32_t> Test_Parallel(maxSchluessel,maxSchluessel,usrGroesseSchluessel,usrGroesseHashtabelle);

    Test_Parallel.erzeuge_schluessel();
    Test_Parallel.erzeuge_werte();

    Zeit::grundStarte();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Keine Kollisonsauflösung
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Parallel.insert_hashtabelle(keine_aufloesung,murmer,false);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Lineare Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Parallel.insert_hashtabelle(linear_aufloesung,murmer,false);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Quadratische Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    Test_Parallel.insert_hashtabelle(quadratisch_aufloesung,murmer,false);
   
    //Fasse Resultate zusammen
    Zeit::grundBeende();
    std::cout << "Gesamtdauer für alle offenen Hashverfahren (in Millisekunden)     : ";
    std::cout << Zeit::getGrundDauer() << std::endl;
    std::cout << std::endl;
    std::cout << "Erfolgreich" << std::endl;

    return 0;
};