#include <iostream>
#include <stdlib.h> 
#include <iomanip>

#include <../include/hashfunktionen.h>
#include <../include/basistest_hashtabelle.h>
#include <../include/hashtabelle.h>
#include <../core/hashtabelle.cpp>

template <typename T1, typename T2>
Basistest_Hashtabelle<T1,T2>::Basistest_Hashtabelle(){
    Hashtabelle<T1,T2> pHashtabelle1;
    Hashtabelle<T1,T2> pHashtabelle2(linear_aufloesung,murmer,20);
    Hashtabelle<T1,T2> pHashtabelle3(quadratisch_aufloesung,murmer,40);
    Hashtabelle<T1,T2> pHashtabelle4(beliebig_aufloesung,murmer,60);

    test_Hashtabelle1 = pHashtabelle1;
    test_Hashtabelle2 = pHashtabelle2; 
    test_Hashtabelle3 = pHashtabelle3; 
    test_Hashtabelle4 = pHashtabelle4;  
};

template <typename T1, typename T2>
Basistest_Hashtabelle<T1,T2>::~Basistest_Hashtabelle(){
};

template <typename T1, typename T2>
void Basistest_Hashtabelle<T1,T2>::testhashtyp(){
    if (test_Hashtabelle1.getHashTyp()!=keine_aufloesung){
        std::cout << "Der Hashtyp für die " << std::quoted("test_Hashtabelle1");
        std::cout << " soll " << std::quoted("keine Auflösung") << " lauten." << std::endl;
        exit (EXIT_FAILURE);
    } 

    if (test_Hashtabelle2.getHashTyp()!=linear_aufloesung){
        std::cout << "Der Hashtyp für die " << std::quoted("test_Hashtabelle2");
        std::cout << " soll " << std::quoted("lineare Auflösung") << " lauten." << std::endl;
        exit (EXIT_FAILURE);
    }
    
    if (test_Hashtabelle3.getHashTyp()!=quadratisch_aufloesung){
        std::cout << "Der Hashtyp für die " << std::quoted("test_Hashtabelle3");
        std::cout << " soll " << std::quoted("quadratische Auflösung") << " lauten." << std::endl;
        exit (EXIT_FAILURE);
    }
    
    if (test_Hashtabelle4.getHashTyp()!=beliebig_aufloesung){
        std::cout << "Der Hashtyp für die " << std::quoted("test_Hashtabelle4");
        std::cout << " soll " << std::quoted("beliebige Auflösung") << " lauten." << std::endl;
        exit (EXIT_FAILURE);
    }
};

template <typename T1, typename T2>
void Basistest_Hashtabelle<T1,T2>::testgroesse(){
    if (test_Hashtabelle1.getGroesseHashtabelle()!=2){
        std::cout << "Die Größe der " << std::quoted("test_Hashtabelle1");
        std::cout << " soll " << std::quoted("2") << " sein." << std::endl;
        exit (EXIT_FAILURE);
    } 

    if (test_Hashtabelle2.getGroesseHashtabelle()!=20){
        std::cout << "Die Größe der " << std::quoted("test_Hashtabelle2");
        std::cout << " soll " << std::quoted("20") << " sein." << std::endl;
        exit (EXIT_FAILURE);
    }

    if (test_Hashtabelle3.getGroesseHashtabelle()!=40){
        std::cout << "Die Größe der " << std::quoted("test_Hashtabelle3");
        std::cout << " soll " << std::quoted("40") << " sein." << std::endl;
        exit (EXIT_FAILURE);
    }

    if (test_Hashtabelle4.getGroesseHashtabelle()!=60){
        std::cout << "Die Größe der " << std::quoted("test_Hashtabelle4");
        std::cout << " soll " << std::quoted("60") << " sein." << std::endl;
        exit (EXIT_FAILURE);
    }
};