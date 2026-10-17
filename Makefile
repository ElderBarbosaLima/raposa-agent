.PHONY: install test lint sync api ui eval

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

sync:
	uv run raposa-sync --season $$(date +%Y)

api:
	uv run uvicorn raposa.api:app --reload

ui:
	uv run streamlit run app/streamlit_app.py

eval:
	uv run python -m evals.run
