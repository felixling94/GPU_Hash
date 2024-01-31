set(TD_I1 ../../../include/hashtabelle.h  ../../../include/dynamisch_hashtabelle.h  ../../../include/deklaration.cuh 
../../../core/hashtabelle.cpp  ../../../core/dynamisch_hashtabelle.cpp  ../../../src/hashtabelle.cuh  
../../../src/dynamisch_hashtabelle.cuh ../../../src/test_main.h  ../../../test/insert/dynamisch/hashtabelle_klein.cu)

# set(TD_I2 ../../../include/hashtabelle.h  ../../../include/dynamisch_hashtabelle.h  ../../../include/deklaration.cuh 
# ../../../core/hashtabelle.cpp  ../../../core/dynamisch_hashtabelle.cpp  ../../../src/hashtabelle.cuh  
# ../../../src/dynamisch_hashtabelle.cuh ../../../src/test_main.h  ../../../test/insert/dynamisch/hashtabelle_gross.cu)

add_executable(testDynamischInsert_klein ${TD_I1})
# add_executable(testDynamischInsert_gross ${TD_I2})