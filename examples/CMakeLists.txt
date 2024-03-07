set(TS_H   ../../examples/example_hash.cu)

set(TS_I1   ../../examples/hash_table_insert1.cu)
set(TS_I2   ../../examples/hash_table_insert2.cu)
set(TS_I3   ../../examples/hash_table_insert3.cu)
set(TS_I4   ../../examples/hash_table_insert4.cu)
set(TS_I5   ../../examples/hash_table_insert5.cu)
set(TS_I6   ../../examples/hash_table_insert6.cu)

set(TS_S1   ../../examples/hash_table_search1.cu)

set(TS_D1   ../../examples/hash_table_delete1.cu)

add_executable(exampleHash ${TS_H})

add_executable(exampleInsert1 ${TS_I1})
add_executable(exampleInsert2 ${TS_I2})
add_executable(exampleInsert3 ${TS_I3})
add_executable(exampleInsert4 ${TS_I4})
add_executable(exampleInsert5 ${TS_I5})
add_executable(exampleInsert6 ${TS_I6})

add_executable(exampleSearch1 ${TS_S1})
add_executable(exampleDelete1 ${TS_D1})
