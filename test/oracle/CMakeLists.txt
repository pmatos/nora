configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_test(NAME oracle
         COMMAND ${CMAKE_BINARY_DIR}/bin/nora-lit ${CMAKE_SOURCE_DIR}/test/oracle -v)

# The normalizer's own unit tests (oracle_datum/oracle_alpha) need no
# `racket` binary, so they run under the default debug preset unconditionally
# rather than gating on NORA_HAVE_RACKET like the `oracle` lit suite above.
add_test(NAME oracle_normalize_unittests
         COMMAND ${Python3_EXECUTABLE} -m unittest discover
                 -s ${CMAKE_SOURCE_DIR}/scripts -p "test_oracle_*.py"
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts)
