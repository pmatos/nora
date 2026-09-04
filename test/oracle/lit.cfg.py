import os
import lit.formats

config.name = "NORA oracle tests"
config.test_format = lit.formats.ShTest(execute_external=False)

config.suffixes = ['.rkt']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.nora_build_root, 'test', 'oracle')

config.environment = dict(os.environ)

# Replace all NORA tools (norac, nora-lit, ...) with their absolute paths.
bin_dir = os.path.join(config.nora_build_root, 'bin')
for tool_file in os.listdir(bin_dir):
    tool_path = config.nora_build_root + '/bin/' + tool_file
    config.substitutions.append((tool_file, tool_path))

if config.nora_have_racket:
    config.available_features.add('racket')
