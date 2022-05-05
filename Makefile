# You can set these variables from the command line.
PYTHON     = python

SRCS =\
     $(wildcard width/*.py)\
     $(wildcard plots/*.py)

TESTS = $(wildcard test/*.py)

.PHONY: all help clean run test

.DEFAULT_GOAL := all

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean      to remove build files"
	@echo "  run        to run files"
	@echo "  test       to run unit tests"
	@echo "  init       to install requirements"

all:	clean	$(SRCS:.py=.pyc)

init:
	pip install -r requirements.txt

run:	$(SRCS)
	$(SRCS)

test:	clean $(TESTS:.py=.pyc)
	$(PYTHON) -m unittest discover

clean:
	@find . -name '*.pyc' -exec rm --force {} +
	@find . -name '*.pyo' -exec rm --force {} +
	@find . -name '*~' -exec rm --force {} +
	@find . -name 'unittest' -exec rm --force {} +

%.pyc: %.py
	$(PYTHON) -m py_compile  $<