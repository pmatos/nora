Each *node.h defined a variant which the nodes that are part of
a certain category. For example valuenodes.h defines the set of nodes
that are value. Also exprnodes.h defines the nodes that are Expr (expressions).

This creates a hierarchy of nodes independent of C++ class hierarchy.

The hierarchy is (names prefixed by + are actual classes)

ASTNodes
|- +Linklet
|- TLNodes
   |- +DefineValues
   |- Expr
      |- +Identifier
      |- +ArithPlus
      |- +Begin
      |- +Begin0
      |- +Application
      |- +SetBang
      |- +IfCond
      |- +LetValues
      |- Value
         |- +Integer
         |- +Values
         |- +Void
         |- +List
         |- +Lambda
         |- +BooleanLiteral