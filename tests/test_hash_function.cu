#include <catch2/catch_test_macros.hpp>

#include <../include/hash_function.cuh>

const size_t table_size1{5}, table_size2{7}, table_size3{11}, table_size4{13}, table_size5{17};  

TEST_CASE("Hashwerte von Divisions-Rest-Methode wird berechnet.","[Modulo_Func]"){
    REQUIRE(modulo_hash<uint32_t>(30,table_size1) == 0);
    REQUIRE(modulo_hash<uint32_t>(30,table_size2) == 2);
    REQUIRE(modulo_hash<uint32_t>(30,table_size3) == 8);
    REQUIRE(modulo_hash<uint32_t>(30,table_size4) == 4);
    REQUIRE(modulo_hash<uint32_t>(30,table_size5) == 13);
};

TEST_CASE("Hashwerte von multiplikativer Methode werden berechnet.","[Multiply_Func]"){
    REQUIRE(multiplication_hash<uint32_t>(30,table_size1) == 2);
    REQUIRE(multiplication_hash<uint32_t>(30,table_size2) == 3);
    REQUIRE(multiplication_hash<uint32_t>(30,table_size3) == 5);
    REQUIRE(multiplication_hash<uint32_t>(30,table_size4) == 7);
    REQUIRE(multiplication_hash<uint32_t>(30,table_size5) == 9);
};

TEST_CASE("Hashwerte von universeller Hashfunktion werden berechnet.","[Universal_Func]"){
    REQUIRE(universal_hash<uint32_t>(30,table_size1,(size_t)2,(size_t)3,(size_t)7) == 0);
    REQUIRE(universal_hash<uint32_t>(30,table_size2,(size_t)2,(size_t)5,(size_t)7) == 2);
    REQUIRE(universal_hash<uint32_t>(30,table_size3,(size_t)7,(size_t)11,(size_t)25) == 10);
    REQUIRE(universal_hash<uint32_t>(30,table_size4,(size_t)7,(size_t)13,(size_t)27) == 7);
    REQUIRE(universal_hash<uint32_t>(30,table_size5,(size_t)11,(size_t)17,(size_t)19) == 5);
};

TEST_CASE("Zwei Werte zwischen vertauscht.","[Swap_Func]"){
    REQUIRE(swapHash<uint32_t>((size_t)2, (size_t)3, (size_t)10) == 2);
    REQUIRE(swapHash<uint32_t>((size_t)5, (size_t)5, (size_t)10) == 10);
    REQUIRE(swapHash<uint32_t>((size_t)20, (size_t)50, (size_t)20) == 20);
    REQUIRE(swapHash<uint32_t>((size_t)20, (size_t)20, (size_t)50) == 50);
    REQUIRE(swapHash<uint32_t>((size_t)20, (size_t)20, (size_t)13) == 13);
};

TEST_CASE("Werte durch quadratische Sondierungsfunktion werden berechnet.","[Probe2_Func]"){
     REQUIRE(getProbe2(2) == 1);
     REQUIRE(getProbe2(3) == -4);
     REQUIRE(getProbe2(4) == 4);
     REQUIRE(getProbe2(6) == 9);
     REQUIRE(getProbe2(7) == -16);
};