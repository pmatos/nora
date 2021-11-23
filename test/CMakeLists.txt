configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

# FIXME: add check target to run tests instead of python check.py at top-level
#add_custom_target(check
#  COMMAND "${CMAKE_CURRENT_BINARY_DIR}"/nora-lit "${CMAKE_CURRENT_BINARY_DIR}" -v)
