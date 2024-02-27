set(TH_1 test_hash_table.cu   test_hash_function.cu)

add_executable(test_program1 ${TH_1})

# These examples use the standard separate compilation
set( SOURCES_IDIOMATIC_EXAMPLES
    030-Asn-Require-Check.cpp
    100-Fix-Section.cpp
    110-Fix-ClassFixture.cpp
    120-Bdd-ScenarioGivenWhenThen.cpp
    210-Evt-EventListeners.cpp
    232-Cfg-CustomMain.cpp
    300-Gen-OwnGenerator.cpp
    301-Gen-MapTypeConversion.cpp
    302-Gen-Table.cpp
    310-Gen-VariablesInGenerators.cpp
    311-Gen-CustomCapture.cpp
)

string( REPLACE ".cpp" "" BASENAMES_IDIOMATIC_EXAMPLES "${SOURCES_IDIOMATIC_EXAMPLES}" )
set( TARGETS_IDIOMATIC_EXAMPLES ${BASENAMES_IDIOMATIC_EXAMPLES} )

foreach( name ${TARGETS_IDIOMATIC_EXAMPLES} )
    add_executable( ${name}
      ${EXAMPLES_DIR}/${name}.cpp )
endforeach()

set(ALL_EXAMPLE_TARGETS
  ${TARGETS_IDIOMATIC_EXAMPLES}
  test_program1
)

foreach( name ${ALL_EXAMPLE_TARGETS} )
    target_link_libraries( ${name} Catch2WithMain )
endforeach()

list(APPEND CATCH_WARNING_TARGETS ${ALL_EXAMPLE_TARGETS})
set(CATCH_WARNING_TARGETS ${CATCH_WARNING_TARGETS} PARENT_SCOPE)
