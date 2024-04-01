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

        operation_type type_operation;

    public:
        Benchmark():upload(0),run(0),download(0),total(0){
        };
        ~Benchmark(){};

        //Gebe den Operationstyp zurück
        operation_type getOperationType(){
            return type_operation;
        };
        
        //Erfasse Daten
        void record(operation_type TypeOperation, 
        float UploadOperation, float RunOperation, float DownloadOperation, float TotalOperation,
        size_t SumFound = 0){
            type_operation = TypeOperation;

            upload = UploadOperation;
            run = RunOperation;
            download = DownloadOperation;
            total = TotalOperation;

            if (TypeOperation == search_hash_table) sum_found = SumFound;
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

        //Drucke Zeitmessung
        void print(){
            if (type_operation == insert_hash_table){
                std::cout << "Speicherung von Schlüsseln in der Hashtabelle" << std::endl;
                std::cout << std::endl;
            }else if(type_operation == search_hash_table){
                std::cout << "Suche nach Schlüsseln in der Hashtabelle" << std::endl;
                std::cout << std::endl;
                std::cout << "Anzahl der gesuchten Zellen                 : ";
                std::cout << sum_found << std::endl;
            }else if(type_operation == delete_hash_table){
                std::cout << "Löschung von Schlüsseln in der Hashtabelle" << std::endl;
                std::cout << std::endl;
            }else{
                std::cout << "Berechnung der Hashwerte" << std::endl;
                std::cout << std::endl;
            }
            std::cout << "Dauer zum Hochladen (ns)                    : ";
            std::cout <<  upload * 1000000 << std::endl;
            std::cout << "Dauer zur Ausführung (ns)                   : ";
            std::cout <<  run * 1000000 << std::endl;
            std::cout << "Dauer zum Herunterladen (ns)                : ";
            std::cout <<  download * 1000000 << std::endl;
            std::cout << "Gesamtdauer (ns)                            : ";
            std::cout <<  total * 1000000 << std::endl;
        };
};

#endif