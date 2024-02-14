#include <iomanip>

#include <../include/basistest_hashtabelle.h>
#include <../core/basistest_hashtabelle.cpp>

void basistest_hashtabelle(){
    Basistest_Hashtabelle<uint32_t,uint32_t> Basistest_Hashtabelle;
    Basistest_Hashtabelle.testgroesse();
    Basistest_Hashtabelle.testhashtyp();

    std::cout << "Basistest 1: Hashtabelle (Erfolgreich)" << std::endl;
};

int main(){
    basistest_hashtabelle();
    std::cout << "Erfolgreich" << std::endl;

    return 0;
}