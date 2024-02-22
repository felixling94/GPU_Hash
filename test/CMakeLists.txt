set(TS_H   ../include/base.h   ../include/declaration.cuh   ../include/hash_function.cuh   
../tools/timer.cuh   ../../test/test_hash.cuh   ../../test/test_hash.cu)

set(TS_I1   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../test/test_hash_table.cuh   ../../test/hash_table_insert1.cu)

set(TS_S1   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../test/test_hash_table.cuh   ../../test/hash_table_search1.cu)

set(TS_D1   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../test/test_hash_table.cuh   ../../test/hash_table_delete1.cu)

add_executable(testHash ${TS_H})
add_executable(testInsert1 ${TS_I1})
add_executable(testSearch1 ${TS_S1})
add_executable(testDelete1 ${TS_D1})