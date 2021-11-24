configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check
  COMMAND "${CMAKE_BINARY_DIR}/bin/nora-lit" "${CMAKE_SOURCE_DIR}/test" -v)
