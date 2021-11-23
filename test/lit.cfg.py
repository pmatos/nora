import lit.formats

config.name = "NORA lit tests"
config.test_format = lit.formats.ShTest(True)

config.suffixes = ['.rkt']

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.nora_src_root, 'out', 'test')

# Replace all NORA tools with their absolute paths
bin_dir = os.path.join(config.nora_build_root, 'bin')
for tool_file in os.listdir(bin_dir):
    tool_path = config.nora_build_root + '/bin/' + tool_file
    config.substitutions.append((tool_file, tool_path))

# Also make the `not` command available
not_file = config.nora_src_root + '/scripts/not.py'
python = sys.executable.replace('\\', '/')
config.substitutions.append(('not', python + ' ' + not_file))
