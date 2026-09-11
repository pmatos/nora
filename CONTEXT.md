# NORA domain glossary

Shared vocabulary for the interpreter. Seeded 2026-09-11 with the **value
register** terms sharpened by the `value-register-take-vs-borrow` deepening; extend
it as further modules are deepened.

## The value register

- **Value register** — the abstract machine's single value slot (`Val` in
  `Interpreter`). Its handle is `class Value` (`src/include/Value.h`), which
  carries at most one of three **representations**: a `Legacy` exclusively-owned
  `unique_ptr<ValueNode>`, a `Shared` `shared_ptr<ValueNode>` aliasing an
  Environment binding, or an `Imm` immediate word.

- **Immediate** — an allocation-free `nr_value` word (booleans, chars, void; M2/GC
  S6+) carried directly by the register, materialized into a `Legacy` node only
  when a consumer needs a `ValueNode*`.

The register exposes three kinds of access, and picking the right one is what
keeps a `Shared` value from being cloned for nothing:

- **Borrow (peek)** — read the held value without consuming it: `Value::get()`
  (untyped `ValueNode*`), `Value::as<T>()` (typed `const T*`, `nullptr` on a
  type mismatch), or, for an engaged immediate specifically, the raw-word fast
  path `isImmediate()`/`rawImmediate()` (e.g. `step(Frame::IfBranch&)`), which
  reads the bare `nr_value` word without even materializing it into `Legacy`.
  Never clones; the node (or word) stays owned by the register.

- **Move into a sink** — hand the whole `Value` to a callee that takes a `Value`
  by value (`deliver`, `envSet`, `Environment::add`). Transfers ownership by
  move; never clones. `Value::toShared()` transfers similarly but is a method
  called *on* the handle rather than a parameter it's moved into — it consumes
  `this` and returns a `shared_ptr<ValueNode>`, not a callee taking a `Value`.

- **Clone-out** — `Value::takeLegacy()`, the only path that clones a `Shared`
  value, producing an exclusively-owned `unique_ptr<ValueNode>`. Reserved for
  splicing a value into a `unique_ptr<ExprNode>` AST-child slot (`ast::Values`,
  `ast::List`).
