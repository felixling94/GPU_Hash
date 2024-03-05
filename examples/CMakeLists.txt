set(TS_H   ../include/base.h   ../include/declaration.cuh   ../include/hash_function.cuh   
../tools/timer.cuh   ../../examples/example_hash.cuh   ../../examples/example_hash.cu)

set(TS_I1   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_insert1.cu)

set(TS_I2   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_insert2.cu)

set(TS_I3   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_insert3.cu)

set(TS_I4   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_insert4.cu)

set(TS_I5   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_insert5.cu)

set(TS_I6   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_insert6.cu)

set(TS_S1   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_search1.cu)

set(TS_D1   ../include/hash_table.h   ../include/base.h   ../include/declaration.cuh
../include/hash_function.cuh   ../core/hash_table.cuh   ../core/hash_method.cuh   
../tools/timer.cuh  ../../examples/example_hash_table.cuh   ../../examples/hash_table_delete1.cu)

add_executable(exampleHash ${TS_H})

add_executable(exampleInsert1 ${TS_I1})
add_executable(exampleInsert2 ${TS_I2})
add_executable(exampleInsert3 ${TS_I3})
add_executable(exampleInsert4 ${TS_I4})
add_executable(exampleInsert5 ${TS_I5})
add_executable(exampleInsert6 ${TS_I6})

add_executable(exampleSearch1 ${TS_S1})
add_executable(exampleDelete1 ${TS_D1})
