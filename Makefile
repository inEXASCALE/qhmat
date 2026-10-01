.PHONY: html clean-docs

html:
	$(MAKE) -C docs html

clean-docs:
	$(MAKE) -C docs clean
