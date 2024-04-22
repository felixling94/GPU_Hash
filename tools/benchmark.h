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

        size_t sum_found;
        size_t num_cells_deleted;

        operation_type type_operation;

        hash_type hashtype;
        hash_function hashfunction;

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
        hash_type HashType = no_probe, size_t SumFound = 0, size_t NumCellsDeleted = 0,
        hash_function HashFunction = modulo){
            type_operation = TypeOperation;

            upload = UploadOperation;
            run = RunOperation;
            download = DownloadOperation;
            total = TotalOperation;

            if (TypeOperation == insert_hash_table || TypeOperation == search_hash_table || TypeOperation == delete_hash_table)
                hashtype = HashType;
            if (TypeOperation == calculate_hash_value)  hashfunction = HashFunction;

            if (TypeOperation == search_hash_table) sum_found = SumFound;
            if (TypeOperation == delete_hash_table) num_cells_deleted = NumCellsDeleted;
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

        //Gebe die Anzahl aller gesuchten Schlüssel zurück
        size_t getSumFound(){
            return sum_found;
        };

        //Gebe die Anzahl aller gelöschten Schlüssel zurück
        size_t getNumCellsDeleted(){
            return num_cells_deleted;
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
                std::cout <<  total  << std::endl;

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
                std::cout <<  total << "," << sum_found << std::endl;
        
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
                std::cout <<  total << "," << num_cells_deleted << std::endl;
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