set(TD_I1 ../../include/hashtabelle.h  ../../include/deklaration.cuh 
../../include/test_main.h  ../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh 
../../src/hashtabelle_kernels_suchen.cuh  ../../src/hashtabelle_kernels_insert.cuh
  ../../test/insert/hashtabelle_klein.cu)

set(TD_I2 ../../include/hashtabelle.h  ../../include/deklaration.cuh  ../../include/test_main.h
../../core/hashtabelle.cpp ../../core/hashtabelle.cuh ../../src/hashtabelle_kernels_suchen.cuh  
../../src/hashtabelle_kernels_insert.cuh ../../test/insert/hashtabelle_gross.cu)

set(TC_I1 ../../include/hashtabelle.h  ../../include/deklaration.cuh  ../../include/test_main.h
../../core/hashtabelle.cpp ../../core/hashtabelle.cuh ../../src/hashtabelle_kernels_suchen.cuh  
../../src/hashtabelle_kernels_insert.cuh ../../test/insert/hashtabelleCuckoo_klein.cu)

set(TC_I2 ../../include/hashtabelle.h  ../../include/deklaration.cuh  ../../include/test_main.h
../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh  ../../src/hashtabelle_kernels_suchen.cuh  
../../src/hashtabelle_kernels_insert.cuh ../../test/insert/hashtabelleCuckoo_gross.cu)

add_executable(testdatenInsert_klein ${TD_I1})
add_executable(testdatenInsert_gross ${TD_I2})
add_executable(testCuckooInsert_klein ${TC_I1})
add_executable(testCuckooInsert_gross ${TC_I2})