#ifndef HASHTABELLE_H
#define HASHTABELLE_H

#include <string>

#include "base.h"
#include <../tools/benchmark.h>

template <typename T>
class Hash_Table{
    private:
        size_t table_size;
        cell<T> * hash_table1;
        cell<T> * hash_table2;

        hash_type type_hash;
        hash_function function1;
        hash_function function2;

        Benchmark * benchmark_hash_table = new Benchmark[3];

        std::string getCell(size_t i, int j = 0);
    
    public:
        Hash_Table();
        Hash_Table(hash_type HashType, hash_function function_1, hash_function function_2, size_t TableSize);
        ~Hash_Table();

        size_t getNumCell();
        size_t getTableSize();
        cell<T> * getTable(int i = 0);

        hash_type getHashType();
        hash_function getHashFunction(int i = 0);

        Benchmark getBenchmark(operation_type type);
        Benchmark * getBenchmarkList();

        void print();
        
        void insert(T key, T key_length);
        void insert_List(T * keyList, T * keyLengthList, int numBlocks, int numThreadsProBlock);
        
        bool search(T key, T key_length);
        void search_List(T * keyList, T * keyLengthList, int numBlocks, int numThreadsProBlock);

        void deleteKey(T key, T key_length);
        void delete_List(T * keyList, T * keyLengthList, int numBlocks, int numThreadsProBlock);
};

#endif