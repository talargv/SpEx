.PHONY: install

install:
	@echo "Installing dependencies..."
	@bash install_dependencies.sh
	@pip install -e .
	@echo "Installation complete."