set(TS_I1 ../../../include/hashtabelle.h  ../../../include/statisch_hashtabelle.h  ../../../include/deklaration.cuh 
../../../core/hashtabelle.cpp  ../../../core/statisch_hashtabelle.cpp  ../../../src/hashtabelle.cuh  
../../../src/statisch_hashtabelle.cuh ../../../src/test_main.h  ../../../test/insert/statisch/hashtabelle_klein.cu)

set(TS_I2 ../../../include/hashtabelle.h  ../../../include/statisch_hashtabelle.h  ../../../include/deklaration.cuh 
../../../core/hashtabelle.cpp  ../../../core/statisch_hashtabelle.cpp  ../../../src/hashtabelle.cuh  
../../../src/statisch_hashtabelle.cuh ../../../src/test_main.h  ../../../test/insert/statisch/hashtabelle_gross.cu)

# set(TC_I1 ../../include/hashtabelle.h  ../../include/deklaration.cuh  ../../core/test_main.h
# ../../core/hashtabelle.cpp ../../core/hashtabelle.cuh ../../src/hashtabelle_kernels_suchen.cuh  
# ../../src/hashtabelle_kernels_insert.cuh ../../test/insert/hashtabelleCuckoo_klein.cu)

# set(TC_I2 ../../include/hashtabelle.h  ../../include/deklaration.cuh  ../../core/test_main.h
# ../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh  ../../src/hashtabelle_kernels_suchen.cuh  
# ../../src/hashtabelle_kernels_insert.cuh ../../test/insert/hashtabelleCuckoo_gross.cu)

add_executable(testStatischInsert_klein ${TS_I1})
add_executable(testStatischInsert_gross ${TS_I2})
# add_executable(testCuckooInsert_klein ${TC_I1})
# add_executable(testCuckooInsert_gross ${TC_I2})