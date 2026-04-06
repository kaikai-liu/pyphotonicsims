
install:
	uv sync --extra dev
	uv run pre-commit install

test:
	uv run pytest

cov:
	uv run pytest --cov=pyphotonicsims

mypy:
	uv run mypy . --ignore-missing-imports

lint:
	uv run flake8

pylint:
	uv run pylint pyphotonicsims

lintd:
	uv run pydocstyle pyphotonicsims

doc8:
	uv run doc8 docs/

update:
	uv lock --upgrade
