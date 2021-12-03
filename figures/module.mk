
TEX_FIGURES:=$(wildcard figures/*.tex)
TEX_FIGURES:=$(filter-out figures/drawparticle.tex, $(TEX_FIGURES))
FIGURES+=$(patsubst %.tex, %.pdf, $(TEX_FIGURES))

-include $(TEXFIGURES:%.tex=%.deps)

figures/%.pdf : figures/%.tex
	cd figures; \
	lualatex $(notdir $<)

