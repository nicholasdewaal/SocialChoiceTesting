
init:
    pip install -r requirements.txt
	python3 ./pref_matrix/setup.py build_ext --inplace

test:
	python3 -m pytest tests
