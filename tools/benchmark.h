#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <iostream>

#include <../include/base.h>

class Benchmark{
    private:
        float upload;
        float run;
        float download;
        float total;

        operation_type type_operation;

        hash_type hashtype;
        hash_function hashfunction;

         size_t num_cells;

    public:
        Benchmark():upload(0),run(0),download(0),total(0){
        };
        ~Benchmark(){};

        //Gebe den Operationstyp zurück
        operation_type getOperationType(){
            return type_operation;
        };

        //Gebe den Hashtyp zurück
        hash_type getHashType(){
            return hashtype;
        };

        //Gebe die Hashfunktion zurück
        hash_function getHashFunction(){
            return hashfunction;
        };

        //Erfasse Daten
        void record(operation_type TypeOperation, 
        float UploadOperation, float RunOperation, float DownloadOperation, float TotalOperation,
        size_t NumCells = 0, hash_type HashType = no_probe, hash_function HashFunction = modulo){
            type_operation = TypeOperation;

            upload = UploadOperation;
            run = RunOperation;
            download = DownloadOperation;
            total = TotalOperation;

            if (TypeOperation == insert_hash_table || TypeOperation == search_hash_table || TypeOperation == delete_hash_table){
                hashtype = HashType;
                num_cells = NumCells;
            }

            if (TypeOperation == calculate_hash_value)  hashfunction = HashFunction;
        };

        //Gebe die Dauer des Hochladens zurück
        float getDurationUpload(){
            return upload;
        };

        //Gebe die Dauer der Ausführung zurück
        float getDurationRun(){
            return run;
        };

        //Gebe die Dauer des Herunterladens zurück
        float getDurationDownload(){
            return download;
        };

        //Gebe die Dauer der Kernelausführung zurück
        float getDurationTotal(){
            return total;
        };

        //Gebe die Anzahl aller gespeicherten, gesuchten und gelöschten Schlüssel zurück
        size_t getNumCells(){
            return num_cells;
        };

        //Drucke Zeitmessung
        void print(){
            std::string HashTypeString;
            
            if (type_operation == insert_hash_table){
                if (hashtype == linear_probe){
                    HashTypeString.append("insert_linear<T>");
                }else if (hashtype == quadratic_probe){
                    HashTypeString.append("insert_quadratic<T>");
                }else if (hashtype == double_probe){
                    HashTypeString.append("insert_double<T>");
                }else if (hashtype == cuckoo_probe){
                    HashTypeString.append("insert_cuckoo<T>");
                }else{
                    HashTypeString.append("insert_normal<T>");
                }
                std::cout << HashTypeString << ", " << upload;
                std::cout << ", " << run << ", " << download  << ", ";
                std::cout <<  total << "," << num_cells << std::endl;

            }else if (type_operation == search_hash_table){
                if (hashtype == linear_probe){
                    HashTypeString.append("search_linear<T>");
                }else if (hashtype == quadratic_probe){
                    HashTypeString.append("search_quadratic<T>");
                }else if (hashtype == double_probe){
                    HashTypeString.append("search_double<T>");
                }else if (hashtype == cuckoo_probe){
                    HashTypeString.append("search_cuckoo<T>");
                }else{
                    HashTypeString.append("search_normal<T>");
                }
                std::cout << HashTypeString << ", " << upload ;
                std::cout << ", " << run << ", " << download  << ", ";
                std::cout <<  total << "," << num_cells << std::endl;
        
            }else if (type_operation == delete_hash_table){
                if (hashtype == linear_probe){
                    HashTypeString.append("delete_linear<T>");
                }else if (hashtype == quadratic_probe){
                    HashTypeString.append("delete_quadratic<T>");
                }else if (hashtype == double_probe){
                    HashTypeString.append("delete_double<T>");
                }else if (hashtype == cuckoo_probe){
                    HashTypeString.append("delete_cuckoo<T>");
                }else{
                    HashTypeString.append("delete_normal<T>");
                }
                std::cout << HashTypeString << ", " << upload;
                std::cout << ", " << run << ", " << download << ", ";
                std::cout <<  total << "," << num_cells << std::endl;
            }else{
                if (hashfunction == universal0 || hashfunction == universal1 ||hashfunction == universal2 || hashfunction == universal3){
                    HashTypeString.append("calculate_universal_hash_kernel<T>");
                }else{
                    HashTypeString.append("calculate_hash_kernel<T>");
                }
                std::cout << HashTypeString << ", " << upload;
                std::cout << ", " << run << ", " << download << ", ";
                std::cout <<  total << std::endl;
            }
        };
};

#endif