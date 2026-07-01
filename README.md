# NORA

NORA is an experimental Racket implementation written with a LLVM backend. 

# Building & Testing

## Prerequisites

- [CMake](https://cmake.org) >= 3.24 and [Ninja](https://ninja-build.org)
- [LLVM](https://llvm.org) 22, built with either its matching Clang or GCC >= 13
- [GMP](https://gmplib.org) including the C++ bindings (`gmpxx`)
- Python 3 with [lit](https://pypi.org/project/lit/) for the integration tests

The default build depends only on LLVM and GMP. The experimental NIR MLIR
dialect is **opt-in** and is not needed to build or run the interpreter — see
[The NIR MLIR dialect](#the-nir-mlir-dialect) below.

## Build and run the tests

The project ships [CMake presets](CMakePresets.json):

```
$ git clone https://github.com/pmatos/nora
$ cd nora
$ cmake --preset debug
$ cmake --build --preset debug
$ ctest --preset debug
```

Other presets are available: `release`, `asan`, `ubsan`, `coverage`, and
`mlir` (builds the opt-in NIR dialect and requires MLIR to be installed).

Run the interpreter on a linklet directly:

```
$ ./build/debug/bin/norac test/integration/arithplus.rkt
2
```

If you modify the project you can run the CI test workflow locally with
[act](https://github.com/nektos/act):

```
$ act -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04 -j test
```

## The NIR MLIR dialect

NORA's planned compiler backend will lower to an MLIR dialect called NIR (Nora
IR; see the [roadmap](#compilation-of-racket)). It is currently an empty
scaffold that the interpreter does not use, so MLIR is opt-in and off by
default. Enable it with `cmake --preset mlir` or `-DNORA_ENABLE_MLIR=ON` once
MLIR is installed.

# Linklets

Everything in Racket land is compiled down into a linklet. The linklet uses the language of [Fully Expanded Programs (FEP)](https://docs.racket-lang.org/reference/syntax-model.html#%28part._fully-expanded%29), therefore to compile Racket one needs an expander, a compiler for FEP, and a FEP runtime. The runtime of FEP is responsible to implement access to OS specific stuff like threads, filesystem, etc (which is what in RacketCS is being done by [Rumble](https://github.com/racket/racket/tree/master/racket/src/cs/rumble)).

## FEP vs Linklets

FEP (Fully Expanded Programs) are not the same as Linklets. According to Matthew Flatt in an email: "The benefit of the layer is that you can more easily arrive at an implementation where you can run the expander itself on the target platform.". For example, Pycket used FEP but transitioned to using linkets in https://github.com/pycket/pycket/pull/232. Racketscript has this ongoing PR to compile to linklets: https://github.com/racketscript/racketscript/pull/266

# Compilation of Racket

Racket is a special language to compile due to its dependency on an expander. This expander is currently [implemented in Racket](https://github.com/racket/racket/tree/master/racket/src/expander) (a few years ago it was in C), and lives in the main [racket](https://github.com/racket/racket) repository. 

Since this expander is written in Racket, it needs to bootstrap itself. In other words, we need a precompiled expander to expand the expander into a linklet so that we can then link that into the compiler to expand user programs. At the moment we are [extracting the racket expander](https://github.com/pmatos/nora-lang/blob/main/expander/expander.rktl) into our repo and until we implement our own, there's no reason we can't use Racket's.

So the MVP plan would be to have something that resembles:

```
      Racket Source --(1)--> expander.rktl (linklet) --------------(4)------------\
                                |                                                 |
User Code ----------------------+(2)-----------> Expanded User Code ---------(5)--|
                                                                                  |
                                                                              NIR Dialect 
                                                                               (MLIR)
                                                                                  |
                                                                                  |(7)
                            Runtime ------------------(3)-------------------> LLVM IR
                                                                                  |
                                                                                 (6)
                                                                                  |
                                                                             JIT/Binary
```

The driver of the compilation is, unexpectedly, the expander. The expander holds the code to read a racket module and compile it. It is the job of NORA to quickstart that by starting to interpret the expander and provide the hooks required by the expander for everything to work.

Steps:

1. [X] We [extract expander.rktl](https://github.com/pmatos/nora-lang/blob/main/.github/workflows/gen-expander.yml) (the expander linklet through bootstrapping) from the Racket sources;
2. [ ] User code is expander using the expander linklet (requires expander interpreter);
3. [ ] Runtime is compiled using LLVM to LLVM IR;
4. [ ] Expander linklet is compiled using LLVM to LLVM IR;
5. [ ] The compiler frontend transforms the s-expr ast into an MLIR dialect called NIR (Nora IR);
6. [ ] LLVM compiles the LLVM IR modules to Binary or we JIT them;
7. [ ] NIR is lowered to LLVM IR;

# Nora Videos

* [Nora Announcement @ Racketfest 2023](https://www.youtube.com/watch?v=8YE6kJXvuaQ)

# Other related projects using MLIR-C
 
* [Brutus](https://github.com/JuliaLabs/brutus/)
  - Suggested by https://github.com/femtomc
* [PyTorch importer](https://github.com/llvm/torch-mlir/tree/main/python/torch_mlir/dialects/torch/importer/jit_ir/csrc)
  - Suggested by https://github.com/stellaraccident
* [Rise Lang](https://rise-lang.org/)
  - Also video here [RISE A functional pattern-based language in MLIR](https://www.youtube.com/watch?v=ZP_Qfr0EuRA)
* [MLIR for Functional Programming (video)](https://www.youtube.com/watch?v=cyMQbZ0B84Q)
* [Clasp](https://github.com/clasp-developers/clasp)
  - Also video: [Common Lisp in LLVM](https://www.youtube.com/watch?v=mbdXeRBbgDM)
