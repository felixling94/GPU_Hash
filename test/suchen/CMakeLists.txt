################################################################################################
#Parallele Ausführung
################################################################################################
set(TS_I1 ../../include/hashtabelle.h   ../../include/deklaration.cuh   ../../include/hashfunktionen.h
../../include/hashfunktionen.cuh   ../../include/test_parallel.cuh  ../../core/hashtabelle.cpp   
../../core/hashtabelle_kernels.cuh   ../../core/hashtabelle_device.cuh   ../../src/hashtabelle.cuh   
../../tools/timer.h   ../../test/suchen/hashtabelle_klein.cu)

set(TS_I2 ../../include/hashtabelle.h   ../../include/deklaration.cuh   ../../include/hashfunktionen.h
../../include/hashfunktionen.cuh   ../../include/test_parallel.cuh   ../../core/hashtabelle.cpp   
../../core/hashtabelle_kernels.cuh   ../../core/hashtabelle_device.cuh   ../../src/hashtabelle.cuh   
../../tools/timer.h   ../../test/suchen/hashtabelle_gross.cu)

add_executable(testSuchen_klein ${TS_I1})
add_executable(testSuchen_gross ${TS_I2})

################################################################################################
#Sequentielle Ausführung
################################################################################################
set(TS_SS1 ../../include/hashtabelle.h   ../../core/hashtabelle.cpp   ../../include/hashfunktionen.h
../../include/test_sequentiell.h   ../../tools/timer.h   ../../test/suchen/hashtabelle_klein.cpp)

set(TS_SS2 ../../include/hashtabelle.h   ../../core/hashtabelle.cpp   ../../include/hashfunktionen.h
../../include/test_sequentiell.h   ../../tools/timer.h   ../../test/suchen/hashtabelle_gross.cpp)

add_executable(testSequentiellSuchen_klein ${TS_SS1})
add_executable(testSequentiellSuchen_gross ${TS_SS2})