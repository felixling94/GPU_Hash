################################################################################
#Kleine Anzahl an Schlüsseln
################################################################################
add_executable(exampleHash ../../examples/example_hash.cu)
add_executable(exampleInsert0A ../../examples/hash_table_insert0A.cu)
add_executable(exampleInsert0B ../../examples/hash_table_insert0B.cu)

add_executable(exampleSearch0 ../../examples/hash_table_search0.cu)

add_executable(exampleDelete0 ../../examples/hash_table_delete0.cu)

################################################################################
#Große Anzahl an Schlüsseln
################################################################################
add_executable(exampleInsert1 ../../examples/hash_table_insert1.cu)
add_executable(exampleInsert2 ../../examples/hash_table_insert2.cu)
add_executable(exampleInsert3 ../../examples/hash_table_insert3.cu)
add_executable(exampleInsert4 ../../examples/hash_table_insert4.cu)

add_executable(exampleSearch1 ../../examples/hash_table_search1.cu)

add_executable(exampleDelete1 ../../examples/hash_table_delete1.cu)
