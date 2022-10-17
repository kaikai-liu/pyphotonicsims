
install: 
	pip install -r requirements.txt --upgrade
	pip install -e .
	pre-commit install

test:
	pytest

cov:
	pytest --cov= pyphotonicsims

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8 

pylint:
	pylint pyphotonicsims

lintd2:
	flake8 --select RST

lintd:
	pydocstyle pyphotonicsims

doc8:
	doc8 docs/

update:
	pur

update2:
	pre-commit autoupdate --bleeding-edge
