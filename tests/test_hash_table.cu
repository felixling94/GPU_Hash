#include <vector>
#include <random>
#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include <../include/base.h>
#include <../core/hash_table.cuh>

//Erzeuge verschiedene Werte für die Schlüssel und Werte zufällig
template <typename T1, typename T2>
std::vector<cell<T1,T2>> createCells(size_t cells_size, int min=0, int max=100){
    std::vector<cell<T1,T2>> cells_vector;
    cells_vector.reserve(cells_size);
    
    std::random_device generator;
    size_t seed = generator();
    std::mt19937 rnd(seed);

    std::uniform_int_distribution<T1> dist1(1,max);
    std::uniform_int_distribution<T2> dist2(min,max);

    for (size_t i = 0; i < cells_size; i++){
        T1 rand1 = dist1(rnd);
        T2 rand2 = dist2(rnd);

        cells_vector.push_back(cell<T1,T2>{rand1,rand2});
    }
    return cells_vector;
};

Hash_Table<uint32_t,uint32_t> hash_table0, hash_table1(no_probe,modulo,modulo,30), 
hash_table2(linear_probe,multiplication,modulo,30), hash_table3(quadratic_probe,murmer,modulo,30), 
hash_table4(double_probe,universal0,modulo,30), hash_table5(double_probe,universal1,multiplication,30),
hash_table6(double_probe,universal2,murmer,30), hash_table7(double_probe,universal3,universal0,30), 
hash_table8(cuckoo_probe,dycuckoo_hash1,universal1,30), hash_table9(cuckoo_probe,dycuckoo_hash2,universal2,30),
hash_table10(cuckoo_probe,dycuckoo_hash3,universal3,30), hash_table11(cuckoo_probe,dycuckoo_hash4,dycuckoo_hash1,30),
hash_table12(cuckoo_probe,dycuckoo_hash5,dycuckoo_hash2,30), hash_table13(cuckoo_probe,dycuckoo_hash1,dycuckoo_hash3,30),
hash_table14(cuckoo_probe,dycuckoo_hash1,dycuckoo_hash4,30), hash_table15(cuckoo_probe,dycuckoo_hash1,dycuckoo_hash5,30);

TEST_CASE("Hashtabellen von verschiedenen Typen werden erstellt.","[Hash_Table_Hash_Type]"){
    REQUIRE(hash_table0.getHashType() == no_probe);
    REQUIRE(hash_table1.getHashType() == no_probe);
    REQUIRE(hash_table2.getHashType() == linear_probe);
    REQUIRE(hash_table3.getHashType() == quadratic_probe);
    REQUIRE(hash_table4.getHashType() == double_probe);
    REQUIRE(hash_table8.getHashType() == cuckoo_probe);
};

TEST_CASE("Hashtabellen mit verschiedenen 1. Hashfunktionen werden erstellt.", "[Hash_Table_Func1]"){
    REQUIRE(hash_table0.getHashFunction() == modulo);
    REQUIRE(hash_table1.getHashFunction() == modulo);
    REQUIRE(hash_table2.getHashFunction() == multiplication);
    REQUIRE(hash_table3.getHashFunction() == murmer);
    REQUIRE(hash_table4.getHashFunction() == universal0);
    REQUIRE(hash_table5.getHashFunction() == universal1);
    REQUIRE(hash_table6.getHashFunction() == universal2);
    REQUIRE(hash_table7.getHashFunction() == universal3);
    REQUIRE(hash_table8.getHashFunction() == dycuckoo_hash1);
    REQUIRE(hash_table9.getHashFunction() == dycuckoo_hash2);
    REQUIRE(hash_table10.getHashFunction() == dycuckoo_hash3);
    REQUIRE(hash_table11.getHashFunction() == dycuckoo_hash4);
    REQUIRE(hash_table12.getHashFunction() == dycuckoo_hash5);
};

TEST_CASE("Hashtabellen mit verschiedenen 2. Hashfunktionen werden erstellt.", "[Hash_Table_Func2]"){
    REQUIRE(hash_table4.getHashFunction(1) == modulo);
    REQUIRE(hash_table5.getHashFunction(1) == multiplication);
    REQUIRE(hash_table6.getHashFunction(1) == murmer);
    REQUIRE(hash_table7.getHashFunction(1) == universal0);
    REQUIRE(hash_table8.getHashFunction(1) == universal1);
    REQUIRE(hash_table9.getHashFunction(1) == universal2);
    REQUIRE(hash_table10.getHashFunction(1) == universal3);
    REQUIRE(hash_table11.getHashFunction(1) == dycuckoo_hash1);
    REQUIRE(hash_table12.getHashFunction(1) == dycuckoo_hash2);
    REQUIRE(hash_table13.getHashFunction(1) == dycuckoo_hash3);
    REQUIRE(hash_table14.getHashFunction(1) == dycuckoo_hash4);
    REQUIRE(hash_table15.getHashFunction(1) == dycuckoo_hash5);
};

TEST_CASE("Hashtabellen von verschiedenen Größen werden erstellt.", "[Hash_Table_Size]"){
    REQUIRE(hash_table0.getTableSize() == 2);
    REQUIRE(hash_table1.getTableSize() == 30);
    REQUIRE(hash_table2.getTableSize() == 30);
    REQUIRE(hash_table3.getTableSize() == 30);
    REQUIRE(hash_table4.getTableSize() == 30);
    REQUIRE(hash_table5.getTableSize() == 30);
    REQUIRE(hash_table6.getTableSize() == 30);
    REQUIRE(hash_table7.getTableSize() == 30);
    REQUIRE(hash_table8.getTableSize() == 60);
    REQUIRE(hash_table9.getTableSize() == 60);
    REQUIRE(hash_table10.getTableSize() == 60);
    REQUIRE(hash_table11.getTableSize() == 60);
    REQUIRE(hash_table12.getTableSize() == 60);
    REQUIRE(hash_table13.getTableSize() == 60);
    REQUIRE(hash_table14.getTableSize() == 60);
    REQUIRE(hash_table15.getTableSize() == 60);
};

TEST_CASE("Hashtabellen mit keinen Zellen werden erstellt.", "[Hash_Table_Num_Cells1]"){
    REQUIRE(hash_table0.getNumCell() == 0);
    REQUIRE(hash_table1.getNumCell() == 0);
    REQUIRE(hash_table2.getNumCell() == 0);
    REQUIRE(hash_table3.getNumCell() == 0);
    REQUIRE(hash_table4.getNumCell() == 0);
    REQUIRE(hash_table5.getNumCell() == 0);
    REQUIRE(hash_table6.getNumCell() == 0);
    REQUIRE(hash_table7.getNumCell() == 0);
    REQUIRE(hash_table8.getNumCell() == 0);
    REQUIRE(hash_table9.getNumCell() == 0);
    REQUIRE(hash_table10.getNumCell() == 0);
    REQUIRE(hash_table11.getNumCell() == 0);
    REQUIRE(hash_table12.getNumCell() == 0);
    REQUIRE(hash_table13.getNumCell() == 0);
    REQUIRE(hash_table14.getNumCell() == 0);
    REQUIRE(hash_table15.getNumCell() == 0);
};

TEST_CASE("Hashtabellen mit keinen Zellen werden erstellt.", "[Hash_Table_Num_Cells2]"){
    std::vector<cell<uint32_t,uint32_t>> cellsVector = createCells<uint32_t,uint32_t>(30,0,30);
    cell<uint32_t,uint32_t> * cells = cellsVector.data();

    for (int i=0; i<30; i++){
        hash_table1.insert(cells[i].key,cells[i].value);
        hash_table2.insert(cells[i].key,cells[i].value);
        hash_table3.insert(cells[i].key,cells[i].value);
        hash_table4.insert(cells[i].key,cells[i].value);
        hash_table8.insert(cells[i].key,cells[i].value);
    }

    REQUIRE(hash_table1.getNumCell() > 0);
    REQUIRE(hash_table2.getNumCell() > 0);
    REQUIRE(hash_table3.getNumCell() > 0);
    REQUIRE(hash_table4.getNumCell() > 0);
    REQUIRE(hash_table8.getNumCell() > 0);
};
















