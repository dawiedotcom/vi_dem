FIGURES:=
MODULES=figures

include $(patsubst %, %/module.mk, $(MODULES))

all : $(FIGURES)


