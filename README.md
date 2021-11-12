# nora-lang

NORA-lang is an experimental Racket implementation written with a LLVM backend. This is an experiment on how we can successfully create a different community around Racket. A community that's not based on a core-team making on the decisions but on a community driven by RFCs and user participation. 

Community-wise there are surely many things to decide but I have been thinking about the technical and this is essential a possibility on how to proceed if one wanted to create an alternative implementation of Racket based on LLVM.

# Linklets

Everything in Racket land is compiled down into a linklet. The linklet uses the language of [Fully Expanded Programs (FEP)](https://docs.racket-lang.org/reference/syntax-model.html#%28part._fully-expanded%29), therefore to compile Racket one needs an expander, a compiler for FEP, and a FEP runtime. The runtime of FEP is responsible to implement access to OS specific stuff like threads, filesystem, etc (which is what in RacketCS is being done by [Rumble](https://github.com/racket/racket/tree/master/racket/src/cs/rumble)).

# Compilation of Racket

Racket is a special language to compile due to its dependency on an expander. This expander is currently [implemented in Racket](https://github.com/racket/racket/tree/master/racket/src/expander) (a few years ago it was in C), and lives in the main [racket](https://github.com/racket/racket) repository. 

Since this expander is written in Racket, it needs to bootstrap itself. In other words, we need a precompiled expander to expand the expander into a linklet so that we can then link that into the compiler to expand user programs. At the moment we are [extracting the racket expander](https://github.com/pmatos/nora-lang/blob/main/expander/expander.rktl) into our repo and until we implement our own, there's no reason we can't use Racket's.

So the MVP plan would be to have something that resembles:

```
      Racket Source --(1)--> expander.rktl (linklet) --------------(4)---------------\
                                |                                                    |
User Code ----------------------+(2)-----------> Expanded User Code (linklet) --(5)--|
                                                                                     |
            Runtime (C/C++/Rust...) ------------------(3)----------------------> LLVM IR --(6)--> JIT/Binary
```

I have a feeling here that the linklet needs to be compiled and linked into the JIT/Binary as well but I am not 100% sure. I think this is required in case the user uses `eval` that triggers macro expansion... but, really, should we provide `eval` in the language? Isn't `eval` evil?

Steps:

1. [X] We [extract expander.rktl](https://github.com/pmatos/nora-lang/blob/main/.github/workflows/gen-expander.yml) (the expander linklet through bootstrapping) from the Racket sources;
2. [ ] User code is expander using the expander linklet;
3. [ ] Runtime is compiled using LLVM to LLVM IR;
4. [ ] Expander linklet is compiled using LLVM to LLVM IR (only needed if we provide `eval`?);
5. [ ] The compiler proper;
6. [ ] LLVM compiles the LLVM IR modules to Binary or we JIT them;

## Expanding user code

TODO

## Runtime

The Runtime needs to compile to LLVM IR, so we can use any language supported by LLVM. C++ is the obvious choice but I think Rust would be an interesting possibility as well.

## Compiling the Expander (do we need `eval`?)

TODO

## Compiler proper

TODO
