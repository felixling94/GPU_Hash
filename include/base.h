#ifndef BASE_H
#define BASE_H

#define LOOP_PERCENTAGE 0
#define BLANK 0

//Operationen in einer Hashtabelle
enum operation_type{calculate_hash_value=0, insert_hash_table, search_hash_table, delete_hash_table};

//Arten der offenen Hashverfahren
//0: Keine Kollisionsauflösung
//1: Lineares Sondieren
//2: Quadratisches Sondieren
//3: Doppelte Hashverfahren
//4: Cuckoo-Hashverfahren
enum hash_type{no_probe=0, linear_probe, quadratic_probe, double_probe, cuckoo_probe};

//Arten von Hashfunktionen
enum hash_function{modulo, multiplication, universal0, universal1, universal2, universal3, murmer, 
                   dycuckoo_hash1, dycuckoo_hash2,dycuckoo_hash3,dycuckoo_hash4,dycuckoo_hash5};

//Ein Paar von einem Schlüssel und dessen Länge
template <typename T>
struct cell{
    T key = BLANK;
    T key_length = BLANK;
};

#endif