"""Test whether the examples are still working."""

import runpy

import pytest


@pytest.mark.examples
class TestExamples:
    def test_ex01(self):
        runpy.run_path("../examples/ex01_ising_singleshot.py")

    def test_ex02(self):
        runpy.run_path("../examples/ex02_ising_temperature_comparison.py")

    def test_ex03(self):
        runpy.run_path("../examples/ex03_ising_multipleMC.py")

    def test_ex04(self):
        runpy.run_path("../examples/ex04_mala_singleshot.py")

    def test_ex05(self):
        runpy.run_path("../examples/ex05_mala_multipleMC.py")

    def test_ex06(self):
        runpy.run_path("../examples/ex06_qe_singleshot.py")

    def test_ex07(self):
        runpy.run_path("../examples/ex07_qe_paralleltempering.py")


