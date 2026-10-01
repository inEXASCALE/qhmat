PYTHON ?= python

.PHONY: html clean-docs

html:
	$(PYTHON) -m mkdocs build --strict

clean-docs:
	rm -rf docs/_build
