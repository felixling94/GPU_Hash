#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <stdint.h>

#include "test_insert.cuh"
#include <../data/datenvorlage.h>
#include <../tools/timer.h>

int main(int argc, char** argv){
    //1. Deklariere die Variablen
    uint32_t groesseHashtabelle;
    double auslastungsfaktor;

    const size_t NX{800}, NY{200}, NZ{200};
    // const size_t NX{9}, NY{9}, NZ{9};
    const size_t matrix_groesse{(NX * NY * NZ) * sizeof(uint32_t)};
    size_t nx_zellen{NX}, ny_zellen{NY}, nz_zellen{NZ};

    int deviceID{0};
    struct cudaDeviceProp eigenschaften;

    if(argc < 2){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    auslastungsfaktor = atof(argv[1]);

    if (auslastungsfaktor == 0){
        std::cout << "Der Auslastungsfaktor der Hashtabelle muss mehr als Null betragen." << std::endl;
        return -1;
    }
  
    cudaSetDevice(deviceID);
	cudaGetDeviceProperties(&eigenschaften, deviceID);

    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;
    std::cout << "Ausgewähltes " << eigenschaften.name << " mit "
              << (eigenschaften.totalGlobalMem/1024)/1024 << "mb VRAM" << std::endl;
    std::cout << "Gesamtgröße von Kernelargumenten: "
              << ((matrix_groesse * 3 + sizeof(uint32_t)) / 1024 / 1024) << "mb\n" << std::endl;

    groesseHashtabelle = (uint32_t) ceil((double) (NX*NY*NZ) / auslastungsfaktor);

    std::cout << "****************************************************************";
    std::cout << "***************" << std::endl;   
    std::cout << "Anzahl der gespeicherten Zellen             : ";
    std::cout << NX * NY * NZ << std::endl;
    std::cout << "Größe der Hashtabelle                       : ";
    std::cout << groesseHashtabelle << std::endl;

    Test_Insert<uint32_t,uint32_t> test_Insert(nx_zellen,ny_zellen,nz_zellen,groesseHashtabelle);
    test_Insert.erzeuge_zellen(1,(int) (NX*NY*NZ*200/100));

    Zeit::grundStarte();

    /////////////////////////////////////////////////////////////////////////////////////////
    //Keine Kollionsauflösung
    /////////////////////////////////////////////////////////////////////////////////////////
    test_Insert.insert_hashtabelle(keine_aufloesung);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Lineare Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    test_Insert.insert_hashtabelle(linear_aufloesung);
    /////////////////////////////////////////////////////////////////////////////////////////
    //Quadratische Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    test_Insert.insert_hashtabelle(quadratisch_aufloesung);
    //Doppelte Hashverfahren
    /////////////////////////////////////////////////////////////////////////////////////////
    test_Insert.insert_hashtabelle(doppelt_aufloesung);

    //Fasse Resultate zusammen
    Zeit::grundBeende();
    std::cout << std::endl;
    std::cout << "Gesamtdauer für alle offenen Hashverfahren  : ";
    std::cout << Zeit::getGrundDauer() << std::endl;
    std::cout << "(in Millisekunden)" << std::endl;
    
    return 0;
};


    // size_t * schluessel = A_schluessel.data();
    // size_t * werte = A_werte.data();

    // std::cout << "A"<< std::endl;
    // for (int i = 0; i<A_schluessel.size(); i++)
    //     std::cout << schluessel[i] << std::endl;

    // std::cout << "B"<< std::endl;
    // for (int i = 0; i<A_werte.size(); i++)
    //     std::cout << werte[i] << std::endl;