set(TS_H   ../../examples/example_hash.cu)

set(TS_I0   ../../examples/hash_table_insert0.cu)
set(TS_I1   ../../examples/hash_table_insert1.cu)
set(TS_I2   ../../examples/hash_table_insert2.cu)
set(TS_I3   ../../examples/hash_table_insert3.cu)

set(TS_S0   ../../examples/hash_table_search0.cu)
set(TS_S1   ../../examples/hash_table_search1.cu)

set(TS_D0   ../../examples/hash_table_delete0.cu)
set(TS_D1   ../../examples/hash_table_delete1.cu)

add_executable(exampleHash ${TS_H})

add_executable(exampleInsert0 ${TS_I0})
add_executable(exampleInsert1 ${TS_I1})
add_executable(exampleInsert2 ${TS_I2})
add_executable(exampleInsert3 ${TS_I3})

add_executable(exampleSearch0 ${TS_S0})
add_executable(exampleSearch1 ${TS_S1})

add_executable(exampleDelete0 ${TS_D0})
add_executable(exampleDelete1 ${TS_D1})
