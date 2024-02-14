################################################################################################
#Parallele Ausführung
################################################################################################
set(TS_I1 ../../include/hashtabelle.h   ../../include/deklaration.cuh   ../../include/hashfunktionen.h   
../../include/hashfunktionen.cuh   ../../include/test_parallel.cuh   ../../core/hashtabelle.cpp
../../core/hashtabelle_kernels.cuh   ../../core/hashtabelle_device.cuh   ../../src/hashtabelle.cuh   
../../tools/timer.h   ../../test/insert/hashtabelle_klein.cu)

set(TS_I2 ../../include/hashtabelle.h   ../../include/deklaration.cuh   ../../include/hashfunktionen.h     
../../include/hashfunktionen.cuh   ../../include/test_parallel.cuh   ../../core/hashtabelle.cpp   
../../core/hashtabelle_kernels.cuh   ../../core/hashtabelle_device.cuh   ../../src/hashtabelle.cuh   
../../tools/timer.h   ../../test/insert/hashtabelle_gross.cu)

add_executable(testInsert_klein ${TS_I1})
add_executable(testInsert_gross ${TS_I2})

################################################################################################
#Sequentielle Ausführung
################################################################################################

set(TS_IS1 ../../include/hashtabelle.h   ../../core/hashtabelle.cpp   ../../include/hashfunktionen.h  
../../include/test_sequentiell.h   ../../tools/timer.h   ../../test/insert/hashtabelle_klein.cpp)

set(TS_IS2 ../../include/hashtabelle.h   ../../core/hashtabelle.cpp   ../../include/hashfunktionen.h
../../include/test_sequentiell.h   ../../tools/timer.h   ../../test/insert/hashtabelle_gross.cpp)

# set(TS_IS3 ../../include/hashtabelle.h   ../../core/hashtabelle.cpp   ../../include/hashfunktionen.h  
# ../../include/cuckoo_hashtabelle.h   ../../core/cuckoo_hashtabelle.cpp   ../../include/test_sequentiell.h   
# ../../tools/timer.h   ../../test/insert/hashtabelleCuckoo_klein.cpp)

# set(TS_IS4 ../../include/hashtabelle.h   ../../core/hashtabelle.cpp   ../../include/hashfunktionen.h  
# ../../include/cuckoo_hashtabelle.h   ../../core/cuckoo_hashtabelle.cpp   ../../include/test_sequentiell.h   
# ../../tools/timer.h   ../../test/insert/hashtabelleCuckoo_gross.cpp)

add_executable(testSequentiellInsert_klein ${TS_IS1})
add_executable(testSequentiellInsert_gross ${TS_IS2})
# add_executable(testSequentiellCuckooInsert_klein ${TS_IS3})