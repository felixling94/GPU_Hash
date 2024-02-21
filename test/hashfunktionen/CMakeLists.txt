set(TH1   ../../include/deklaration.cuh   ../../hashfunktionen/hashfunktionen.h   ../../hashfunktionen/hashfunktionen.cuh   
../../data/datenvorlage.h   ../../tools/timer.h   ../../test/hashfunktionen/hashfunktion1.cu)

set(TH2   ../../include/deklaration.cuh   ../../hashfunktionen/hashfunktionen.h   ../../hashfunktionen/hashfunktionen.cuh   
../../data/datenvorlage.h   ../../tools/timer.h   ../../test/hashfunktionen/hashfunktion2.cu)

set(TH3   ../../include/deklaration.cuh   ../../hashfunktionen/dycuckoo_funktionen.h   ../../hashfunktionen/dycuckoo_funktionen.cuh   
../../data/datenvorlage.h   ../../tools/timer.h   ../../test/hashfunktionen/hashfunktion3.cu)

add_executable(testhash1 ${TH1})
add_executable(testhash2 ${TH2})
add_executable(testhash3 ${TH3})