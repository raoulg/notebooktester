# tests/helpers/notebook_creator.py

from pathlib import Path

import nbformat


class NotebookCreator:
    @staticmethod
    def create_notebook(cells: list, filename: str, output_dir: Path) -> Path:
        """Create a notebook with given cells and save it to output_dir"""
        nb = nbformat.v4.new_notebook()
        nb.cells.extend([nbformat.v4.new_code_cell(cell) for cell in cells])

        output_path = output_dir / filename
        with open(output_path, "w") as f:
            nbformat.write(nb, f)

        return output_path

    @classmethod
    def create_basic_notebook(cls, output_dir: Path) -> Path:
        """Create a simple test notebook"""
        cells = ["print('Hello, World!')", "2 + 2"]
        return cls.create_notebook(cells, "basic.ipynb", output_dir)

    @classmethod
    def create_timeout_notebook(cls, output_dir: Path) -> Path:
        """Create a notebook that will timeout"""
        cells = ["import time\ntime.sleep(2)"]
        return cls.create_notebook(cells, "timeout.ipynb", output_dir)
