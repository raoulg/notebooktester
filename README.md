# 📓 NotebookTester
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![image](https://shields.io/pypi/v/notebooktester.svg)](https://pypi.org/project/notebooktester/)

A fast, reliable, and parallel Jupyter notebook testing tool with caching support! Test your notebooks with confidence.

## 🌟 Features

- 🚀 **Parallel Execution**: Test multiple notebooks simultaneously
- ⏱️ **Configurable Timeouts**: Set custom timeouts for notebook execution
- 💾 **Smart Caching**: Cache test results to avoid unnecessary re-runs
- 🎯 **Selective Testing**: Only test modified notebooks
- 📊 **Detailed Logging**: Comprehensive logs with test results and errors
- 🛠️ **CLI Support**: Easy to use command-line interface

## 🔧 Installation

Choose your preferred package manager:

### Using pip (slow legacy)

```bash
pip install notebooktester
```

### Using uv (10-100x faster)
see [uv docs](https://docs.astral.sh/uv/) for more info.

```bash
uv add notebooktester
```

## 🚀 Quick Start

Test a single notebook:

```bash
notebooktester path/to/your/notebook.ipynb
```

Test all notebooks in a directory:

```bash
notebooktester path/to/notebooks/directory
```

## 🎮 Command Line Options

```bash
notebooktester [OPTIONS] PATH
```

Options:
- `-t, --timeout SECONDS`: Timeout in seconds for each notebook (default: 60)
- `-w, --workers NUMBER`: Number of parallel workers (default: CPU count)
- `-c, --cache-dir PATH`: Cache directory for test results (default: .notebookcache)
- `-v, --verbose`: Enable verbose output
- `-f, --force`: Ignore cache and force test execution

## 📋 Example Usage

Basic usage:
```bash
notebooktester notebooks/
```

With custom timeout and workers:
```bash
notebooktester notebooks/ -t 120 -w 4
```

Force re-run all tests:
```bash
notebooktester notebooks/ --force
```

## 🔍 Cache Behavior

NotebookTester maintains a cache of test results to optimize performance:
- Only notebooks modified since their last test run are re-tested
- If the notebook timed out, and the current timeout has not been increased, notebook is skipped
- Cached results include success/failure status and error messages
- Force flag (`-f`) bypasses the cache

## 📊 Output

The tool provides:
- Progress bar for test execution
- Colored console output for test results
- Detailed logs in the `logs/` directory
- Summary of passed and failed tests

Example output:
```
>>notebooktester notebooks/ -t 120 -w 4
Running tests with 4 workers
Starting notebook tests - found 10 notebooks
✅ PASSED - notebook1.ipynb: Success
❌ FAILED - notebook2.ipynb: Cell execution error
⏰ TIMEOUT - notebook3.ipynb: A cell timed out

Test Summary: 8 passed, 2 failed
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 👤 Author

Raoul Grouls (Raoul.Grouls@han.nl)
