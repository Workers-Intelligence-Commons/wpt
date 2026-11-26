"""Read text: next-token prediction on windows of the corpus's texts of
kind plain, the windows growing stage by stage, under by-part on its multi-step schedule
(docs/training.md, "The read-text step"). An order that stops after it
makes a base model."""

METHOD = "next-token"
ENVIRONMENT = "text-windows"
KINDS = ("plain",)
OPTIMIZER = {"name": "by-part"}
LOCKED = ()
