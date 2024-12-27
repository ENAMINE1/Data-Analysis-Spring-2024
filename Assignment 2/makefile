VENV = .eda

all: setup run

install-venv:
	@dpkg -s python3-venv >/dev/null 2>&1 || { echo "python3-venv not found. Installing..."; sudo apt update && sudo apt install -y python3-venv; }

setup: install-venv
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

run: setup
	$(VENV)/bin/python Solution.py

clean:
	rm -f *.png

clean-env:
	rm -rf $(VENV)
