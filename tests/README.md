# unit test for PyTorch
reference: test_binary.py

Test examples:
```bash
pytest -s ./
pytest -s test_binary.py
pytest -s test_bainary.py::test_add
```

## Note
- filename must begin with "test_"
- function name must begin with "test_"
- please add tests as much as possible, including scalar/zero-elements/multi-dimension/multiple data types.
