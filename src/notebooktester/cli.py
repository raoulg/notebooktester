from pathlib import Path

from .main import NotebookTester


def run_tests():
    tester = NotebookTester(dir=Path("notebooks"), timeout=60)
    tester.run_tests()
