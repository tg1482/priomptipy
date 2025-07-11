.PHONY: help install test build clean publish

# Default target
help:
	@echo "PriomptiPy - Available commands:"
	@echo ""
	@echo "  install     Install package with dev dependencies"
	@echo "  test        Run tests"
	@echo "  build       Build package for distribution"
	@echo "  clean       Clean build artifacts"
	@echo "  publish     Publish to PyPI"

# Development setup
install:
	uv sync --all-extras
	@echo "✅ Development environment ready"

# Testing
test: install
	uv run pytest tests/test.py -v
	@echo "✅ Tests completed"

# Build package
build: clean
	uv build
	@echo "✅ Package built successfully"
	@echo "📦 Distribution files:"
	@ls -la dist/

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned build artifacts"

# Publishing
publish: build
	@if [ -z "$$PYPI_TOKEN" ]; then \
		echo "❌ Set PYPI_TOKEN: export PYPI_TOKEN=your-token"; \
		exit 1; \
	fi
	uv publish --token $$PYPI_TOKEN
	@echo "✅ Published to PyPI" 