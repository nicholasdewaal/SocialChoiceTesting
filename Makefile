
init:
    pip install -r requirements.txt
	python3 ./pref_matrix/setup.py build_ext --inplace

test:
    py.test tests
