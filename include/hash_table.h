#ifndef HASHTABELLE_H
#define HASHTABELLE_H

#include <../include/base.h>
#include <../tools/benchmark.h>

template <typename T1 = uint32_t, typename T2 = uint32_t>
class Hash_Table{
    private:
        size_t table_size;
        cell<T1,T2> * hash_table1;
        cell<T1,T2> * hash_table2;

        hash_type type_hash;
        hash_function function1;
        hash_function function2;

        kernel_dimension dimension_kernel;
        Benchmark * benchmark_hash_table = new Benchmark[3];

        std::string getCell(size_t i, int j = 0);
    
    public:
        Hash_Table();
        Hash_Table(hash_type HashType, hash_function function_1, hash_function function_2, size_t TableSize,
                   int NumBlocks = 0, int NumThreadsPerBlock = 0);
        ~Hash_Table();

        size_t getNumCell();
        size_t getTableSize();
        cell<T1,T2> * getTable(int i = 0);

        hash_type getHashType();
        hash_function getHashFunction(int i = 0);

        kernel_dimension getKernelDimension();
        Benchmark getBenchmark(operation_type type);
        Benchmark * getBenchmarkList();

        void print();
        
        void insert(T1 key, T2 value);
        void insert_List(T1 * keyList, T2 * valueList, size_t cellSize);
        
        bool search(T1 key, T2 value);
        void search_List(T1 * keyList, T2 * valueList, size_t cellSize);

        void deleteKey(T1 key, T2 value);
        void delete_List(T1 * keyList, T2 * valueList, size_t cellSize);
};

#endif