#include <cstdlib>
#include <iostream>
#include <cmath>

#include <../include/hashtabelle.h>

template <typename T1>
Hashtabelle<T1>::Hashtabelle(){
    groesseHashtabelle = 2;
};

template <typename T1>
Hashtabelle<T1>::Hashtabelle(size_t pGroesse){
    groesseHashtabelle = pGroesse;
};

template <typename T1>
Hashtabelle<T1>::~Hashtabelle(){
};

template <typename T1>
size_t Hashtabelle<T1>::getGroesseHashtabelle(){
    return groesseHashtabelle;
};

//Berechne den Modulo eines Schlüssels
template <typename T1>
size_t Hashtabelle<T1>::getHashwert(T1 pSchluessel){
    return (size_t) pSchluessel % groesseHashtabelle;
};

// template <typename T0, typename T1, typename T2>
// Hashtabelle<T0,T1,T2>::Hashtabelle():
// hashtyp_kode(keine_aufloesung),groesseHashtabelle(2),standDruecke(false){
//     hashtabelle = new Knoten<T1,T2>[2];
//     hashtabelle[0].schluessel = 0;
//     hashtabelle[0].wert = 0;
//     hashtabelle[1].schluessel = 0;
//     hashtabelle[1].wert = 0;
// };

// template <typename T0, typename T1, typename T2>
// Hashtabelle<T0,T1,T2>::Hashtabelle(hashtyp pHashtyp, T0 pGroesse):
// hashtyp_kode(pHashtyp),groesseHashtabelle(pGroesse),standDruecke(false){
//     if (pHashtyp !=keine_aufloesung && pHashtyp!=linear_aufloesung &&
//         pHashtyp !=quadratisch_aufloesung && pHashtyp != cuckoo_aufloesung){
//         std::cout << "Der Kode einer Art der Hashverfahren muss mindestens 0 und weniger als ";
//         std::cout << "4 betragen." << std::endl;
//         exit(-1);
//     }

//     hashtabelle = new Knoten<T1,T2>[pGroesse];
//     for (T0 i = 0; i<pGroesse; i++) {
//         hashtabelle[i].schluessel = 0;
//         hashtabelle[i].wert = 0;
//     }
// };

// template <typename T0, typename T1, typename T2>
// Hashtabelle<T0,T1,T2>::~Hashtabelle(){
// };

// template <typename T0, typename T1, typename T2>
// T0 Hashtabelle<T0,T1,T2>::getGroesseHashtabelle(){
//     return groesseHashtabelle;
// };

// template <typename T0, typename T1, typename T2>
// Knoten<T1,T2>* Hashtabelle<T0,T1,T2>::getHashtabelle(){
//     return hashtabelle;
// };

// template <typename T0, typename T1, typename T2>
// bool Hashtabelle<T0,T1,T2>::getStandDruecke(){
//     return standDruecke;
// };

// template <typename T0, typename T1, typename T2>
// void Hashtabelle<T0,T1,T2>::schalteStandDruecke(){
//     if (standDruecke == false){
//         standDruecke = true;
//     }else{
//         standDruecke = false;
//     }
// };

// template <typename T0, typename T1, typename T2>
// void Hashtabelle<T0,T1,T2>::druckeHashtabelle(){
//     if (standDruecke == true){
//         druckeHashtabelle1(hashtabelle);
//     }else{
//         return;
//     }
// };

// template <typename T0, typename T1, typename T2>
// void Hashtabelle<T0,T1,T2>::druckeHashtabelle1(Knoten<T1,T2> * pHashtabelle){
//     if(groesseHashtabelle>0)
//         std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << std::endl;
    
//     for (T0 i = 0; i<groesseHashtabelle; i++){
//         std::cout << i << "  " << pHashtabelle[i].schluessel << "  " << pHashtabelle[i].wert;
//         std::cout << "" << std::endl;
//     }
// };

// // template <typename T0, typename T1, typename T2>
// // void Hashtabelle<T0,T1,T2>::druckeKollision1(Kollision<T0,T1,T2> * pKollision){
// //     if (standDruecke == true){
// //         std::cout << "Index" << "  " << "Schlüssel" << "  " << "Wert" << "  ";
// //         std::cout << "Zahl der Kollisionen" << std::endl;
        
// //         for (T0 i = 0; i<groesseHashtabelle; i++){
// //             std::cout << i << "  " << pKollision[i].knoten.schluessel << "  ";
// //             std::cout << pKollision[i].knoten.wert << "  " << pKollision[i].zahlKollision  << std::endl;
// //         }
// //     }
// // }

// //Modulo Hash
// template <typename T0, typename T1, typename T2>
// T0 Hashtabelle<T0,T1,T2>::hashwert(T1 pSchluessel){
//     return (T0) pSchluessel % groesseHashtabelle;
// };

// template <typename T0, typename T1, typename T2>
// void Hashtabelle<T0,T1,T2>::insert(Knoten<T1,T2> pKnoten){
//     //Ohne Kollisionsauflösung
//     if (hashtyp_kode == keine_aufloesung){
//         T0 neuIndex = hashwert(pKnoten.schluessel);

//         if(hashtabelle[neuIndex] == 0){
//             hashtabelle[neuIndex] = pKnoten.schluessel;
//             return;
//         }
    
//         if(standDruecke == true){
//             std::cout << "Das Element " << pKnoten.schluessel<< " mit dem Index " << neuIndex << " kann ";
//             std::cout << "der Hashtabelle ohne Kollisionsauflösung nicht hinzugefügt werden." << std::endl;
//             return;
//         }

//     return;

//     //Lineare Hashverfahren
//     }else if(hashtyp_kode == linear_aufloesung){
//         T0 j, neuIndex;

//         j = 0;
//         neuIndex = hashwert(pKnoten.schluessel);
        
//         while(j<groesseHashtabelle){
//             neuIndex = (neuIndex + j) % groesseHashtabelle;

//             if(hashtabelle[neuIndex] == 0){
//                 hashtabelle[neuIndex] = pKnoten.schluessel;
//                 return;
//             }
            
//             ++j;
//         }

//         if(standDruecke == true){
//             std::cout << "Das Element " << pKnoten.schluessel << " mit dem Index " << neuIndex << " kann ";
//             std::cout << "der Hashtabelle bei linearen Hashverfahren nicht hinzugefügt werden." << std::endl;
//             return;
//         }

//     return;

//     //Quadratische Hashverfahren
//     }else if(hashtyp_kode== quadratisch_aufloesung){
//         T0 j, neuIndex,neuIndex2;

//         j = 0;
//         neuIndex = hashwert(pKnoten.schluessel);

//         while (j/2<groesseHashtabelle){
//             neuIndex2 = (T0) (pow(ceil(j/2),2.0)* pow(-1.0, (double)j));
//             neuIndex = (neuIndex + neuIndex2) % groesseHashtabelle;
        
//             if (hashtabelle[neuIndex] == 0){
//                 hashtabelle[neuIndex] = pKnoten.schluessel;
//                 return;
//             }
      
//             ++j;
//         }

//         if(standDruecke == true){
//             std::cout << "Das Element " << pKnoten.schluessel << " mit dem Index " << neuIndex << " kann ";
//             std::cout << "der Hashtabelle bei quadratischen Hashverfahren nicht hinzugefügt werden." << std::endl;
//             return;
//         }
//         return;

//     //Cuckoo-Hashverfahren
//     }else{
//         //TODO
//         return;
//     }   
// };

// template <typename T0, typename T1, typename T2>
// bool Hashtabelle<T0,T1,T2>::suchen(Knoten<T1,T2> pKnoten){
//     //Ohne Kollisionsauflösung
//     if (hashtyp_kode == keine_aufloesung){
//         T0 neuIndex = hashwert(pKnoten.schluessel);
        
//         if(hashtabelle[neuIndex] == pKnoten.schluessel) return true;
//         return false;

//     //Lineare Hashverfahren
//     }else if(hashtyp_kode == linear_aufloesung){
//         T0 j, neuIndex;

//         j = 0;
//         neuIndex = hashwert(pKnoten.schluessel);
        
//         while(j<groesseHashtabelle){
//             neuIndex = hashwert(neuIndex + j);

//             if(hashtabelle[neuIndex] == pKnoten.schluessel) return true;

//             ++j;
//         }
        
//         return false;

//     //Quadratische Hashverfahren
//     }else if(hashtyp_kode == quadratisch_aufloesung){
//         T0 j, neuIndex,neuIndex2;

//         j = 0;
//         neuIndex = hashwert(pKnoten.schluessel);

//         while (j/2<groesseHashtabelle){
//             neuIndex2 = (T0) (pow(ceil(j/2),2.0)* pow(-1.0, (double)j));
//             neuIndex = hashwert(neuIndex + neuIndex2);
        
//             if (hashtabelle[neuIndex] == pKnoten.schluessel) return true;
      
//             ++j;
//         }

//         return false;

//     //Cuckoo-Hashverfahren
//     }else{
//         //TODO
//         return false;
//     }   
// };
