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
    def create_ray_tune_notebook(cls, output_dir: Path) -> Path:
        """Create a notebook with Ray Tune"""
        cells = [
            """
import ray
from ray import tune
import time

def training_function(config):
    for step in range(config["steps"]):
        time.sleep(0.1)
        tune.report({"score":step})

ray.init()
            """,
            """
analysis = tune.run(
    training_function,
    config={"steps": 5},
    num_samples=2
)
            """,
        ]
        return cls.create_notebook(cells, "ray_tune.ipynb", output_dir)

    @classmethod
    def create_ray_tune_notebook_slow(cls, output_dir: Path) -> Path:
        """Create a too slow notebook with Ray Tune"""
        cells = [
            """
import ray
from ray import tune
import time

def training_function(config):
    for step in range(config["steps"]):
        time.sleep(2)
        tune.report({"score":step})

ray.init()
            """,
            """
analysis = tune.run(
    training_function,
    config={"steps": 50},
    num_samples=2
)
            """,
        ]
        return cls.create_notebook(cells, "ray_tune_slow.ipynb", output_dir)

    @classmethod
    def create_timeout_notebook(cls, output_dir: Path) -> Path:
        """Create a notebook that will timeout"""
        cells = ["import time\ntime.sleep(10)"]
        return cls.create_notebook(cells, "timeout.ipynb", output_dir)
