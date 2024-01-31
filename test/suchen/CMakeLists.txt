# set(TS_S1 ../../include/hashtabelle.h  ../../include/statisch_hashtabelle.h  ../../include/deklaration.cuh 
# ../../core/hashtabelle.cpp  ../../core/statisch_hashtabelle.cpp  ../../core/hashfunktionen.cu
# ../../src/test_main.cpp  ../../src/hashtabelle.cu  ../../src/statisch_hashtabelle.cu
# ../../test/suchen/hashtabelleStatischSuchen_klein.cu)

# set(TD_S1 ../../include/hashtabelle.h  ../../include/deklaration.cuh 
# ../../core/test_main.h  ../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh 
# ../../src/hashtabelle_kernels_suchen.cuh  ../../src/hashtabelle_kernels_insert.cuh
# ../../test/suchen/hashtabelleSuchen_klein.cu)

# set(TD_S2 ../../include/hashtabelle.h  ../../include/deklaration.cuh 
# ../../core/test_main.h  ../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh 
# ../../src/hashtabelle_kernels_suchen.cuh  ../../src/hashtabelle_kernels_insert.cuh
# ../../test/suchen/hashtabelleSuchen_gross.cu)

# set(TC_S1 ../../include/hashtabelle.h  ../../include/deklaration.cuh 
# ../../core/test_main.h  ../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh 
# ../../src/hashtabelle_kernels_suchen.cuh  ../../src/hashtabelle_kernels_insert.cuh 
# ../../test/suchen/hashtabelleSuchenCuckoo_klein.cu)

# set(TC_S2 ../../include/hashtabelle.h  ../../include/deklaration.cuh 
# ../../core/test_main.h  ../../core/hashtabelle.cpp  ../../core/hashtabelle.cuh 
# ../../src/hashtabelle_kernels_suchen.cuh  ../../src/hashtabelle_kernels_insert.cuh 
# ../../test/suchen/hashtabelleSuchenCuckoo_gross.cu)

# add_executable(testStatischSuchen_klein ${TS_S1})
# add_executable(testStatischSuchen_gross ${TS_S2})
# add_executable(testCuckooSuchen_klein ${TC_S1})
# add_executable(testCuckooSuchen_gross ${TC_S2})