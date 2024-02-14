#ifndef BASISTEST_HASHTABELLE_H
#define BASISTEST_HASHTABELLE_H

#include <../include/hashtabelle.h>

template <typename T1, typename T2>
class Basistest_Hashtabelle{
  private:
    Hashtabelle<T1,T2> test_Hashtabelle1;
    Hashtabelle<T1,T2> test_Hashtabelle2;
    Hashtabelle<T1,T2> test_Hashtabelle3;
    Hashtabelle<T1,T2> test_Hashtabelle4;
    
  public:
    Basistest_Hashtabelle();
    ~Basistest_Hashtabelle();
    void testhashtyp();
    void testgroesse();
};

#endif