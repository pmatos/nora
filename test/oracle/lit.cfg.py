import os
import sys
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

# %racket-expand-* substitutions must be registered before %racket below:
# lit applies substitutions in list order via unanchored regex substitution,
# and "%racket" is a literal prefix of "%racket-expand", so registering
# %racket first would mangle "%racket-expand" into "<racket path> -expand".
oracle_expand = os.path.join(config.nora_src_root, 'scripts', 'oracle-expand.rkt')
config.substitutions.append(
    ('%racket-expand', config.nora_racket_executable + ' ' + oracle_expand))

# %racket runs a fixture through the pinned racket via oracle-eval.rkt;
# %normalize is the printed-output normalizer both engines' output is piped
# through before diffing (scaffolded here, see scripts/oracle-normalize.py).
oracle_eval = os.path.join(config.nora_src_root, 'scripts', 'oracle-eval.rkt')
config.substitutions.append(
    ('%racket', config.nora_racket_executable + ' ' + oracle_eval))

normalize = os.path.join(config.nora_src_root, 'scripts', 'oracle-normalize.py')
config.substitutions.append(('%normalize', sys.executable + ' ' + normalize))

# %observe is the observational-equivalence fallback (oracle-observe.rkt)
# for fixtures where structural normalization cannot possibly reconcile
# two hygienically-distinct, same-printed-name binders.
oracle_observe = os.path.join(config.nora_src_root, 'scripts', 'oracle-observe.rkt')
config.substitutions.append(
    ('%observe', config.nora_racket_executable + ' ' + oracle_observe))

if config.nora_have_racket:
    config.available_features.add('racket')
