# -*- coding: utf-8 -*-
"""
:Author: Dominic Hunt
"""

import pytest

import numpy as np
import ndtest.ndtest as ndt


class TestClass_ks2d2s:
    def test_ks2d2s_basic(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((10))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        actual = ndt.ks2d2s(x1, y1, x2, y2)
        expected = 0.3
        assert actual == pytest.approx(expected, rel=1e-1)

    def test_ks2d2s_basic_large(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = np.random.random((1000))
        y2 = np.random.random((1000))
        y1 = np.random.random((1000))
        actual = ndt.ks2d2s(x1, y1, x2, y2)
        expected = 0.12
        assert actual == pytest.approx(expected, rel=1e-2)

    def test_ks2d2s_similar(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = x1.copy()
        y1 = np.random.random((1000))
        y2 = y1.copy()
        actual = ndt.ks2d2s(x1, y1, x2, y2)
        expected = 1.0
        assert actual == pytest.approx(expected, rel=1e-2)

    def test_ks2d2s_equal(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.ks2d2s(x1, y1, x2, y2)
        expected = 1.0
        assert actual == pytest.approx(expected, rel=1e-1)

    def test_ks2d2s_equal_unity(self):
        x1 = np.ones((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.ks2d2s(x1, y1, x2, y2)
        assert np.isnan(actual)

    def test_ks2d2s_equal_zeros(self):
        x1 = np.zeros((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.ks2d2s(x1, y1, x2, y2)
        assert np.isnan(actual)

    def test_ks2d2s_misshaped(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((9))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        with pytest.raises(AssertionError):
            actual = ndt.ks2d2s(x1, y1, x2, y2)

    def test_ks2d2s_boot_similar(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = x1.copy()
        y1 = np.random.random((1000))
        y2 = y1.copy()
        actual = ndt.ks2d2s(x1, y1, x2, y2, nboot=10)
        expected = 1.0
        assert actual == pytest.approx(expected, rel=1e-2)

    def test_ks2d2s_boot_basic(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((10))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        actual = ndt.ks2d2s(x1, y1, x2, y2, nboot=10)
        expected = 0.4
        assert actual == pytest.approx(expected, rel=1e-2)

    def test_ks2d2s_extra_similar(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = x1.copy()
        y1 = np.random.random((1000))
        y2 = y1.copy()
        actual_p, actual_D = ndt.ks2d2s(x1, y1, x2, y2, extra=True)
        expected_D = 0.001
        expected_p = 1.0
        assert actual_D == pytest.approx(expected_D, rel=1e-4)
        assert actual_p == pytest.approx(expected_p, rel=1e-1)

    def test_ks2d2s_extra_basic(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((10))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        actual_p, actual_D = ndt.ks2d2s(x1, y1, x2, y2, extra=True)
        expected_D = 0.4
        expected_p = 0.295
        assert actual_D == pytest.approx(expected_D, rel=1e-2)
        assert actual_p == pytest.approx(expected_p, rel=1e-4)


class TestClass_avgmaxdist:
    def test_avgmaxdist_basic(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((10))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.4
        assert actual == pytest.approx(expected, rel=1e-1)

    def test_avgmaxdist_basic_large(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = np.random.random((1000))
        y2 = np.random.random((1000))
        y1 = np.random.random((1000))
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.0645
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_avgmaxdist_similar(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = x1.copy()
        y1 = np.random.random((1000))
        y2 = y1.copy()
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.001
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_avgmaxdist_equal(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.1
        assert actual == expected

    def test_avgmaxdist_equal_unity(self):
        x1 = np.ones((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.1
        assert actual == expected

    def test_avgmaxdist_equal_unity_large(self):
        x1 = np.ones((1000))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.001
        assert actual == expected

    def test_avgmaxdist_equal_zeros(self):
        x1 = np.zeros((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.1
        assert actual == expected

    def test_avgmaxdist_equal_zeros_large(self):
        x1 = np.zeros((1000))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.avgmaxdist(x1, y1, x2, y2)
        expected = 0.001
        assert actual == expected

    def test_avgmaxdist_misshaped(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((9))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        with pytest.raises(ValueError):#, match='operands could not be broadcast together with shapes (9,1) (10,1) '):
            actual = ndt.avgmaxdist(x1, y1, x2, y2)


class TestClass_maxdist:
    def test_maxdist_basic(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((10))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.4
        assert actual == expected

    def test_maxdist_basic_large(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = np.random.random((1000))
        y2 = np.random.random((1000))
        y1 = np.random.random((1000))
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.065
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_maxdist_similar(self):
        np.random.seed(100)
        x1 = np.random.random((1000))
        x2 = x1.copy()
        y1 = np.random.random((1000))
        y2 = y1.copy()
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.001
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_maxdist_equal(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.1
        assert actual == expected

    def test_maxdist_equal_unity(self):
        x1 = np.ones((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.1
        assert actual == expected

    def test_maxdist_equal_unity_large(self):
        x1 = np.ones((1000))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.001
        assert actual == expected

    def test_maxdist_equal_zeros(self):
        x1 = np.zeros((10))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.1
        assert actual == expected

    def test_maxdist_equal_zeros_large(self):
        x1 = np.zeros((1000))
        x2 = x1.copy()
        y2 = x1.copy()
        y1 = x1.copy()
        actual = ndt.maxdist(x1, y1, x2, y2)
        expected = 0.001
        assert actual == expected

    def test_maxdist_misshaped(self):
        np.random.seed(100)
        x1 = np.random.random((10))
        x2 = np.random.random((9))
        y2 = np.random.random((10))
        y1 = np.random.random((10))
        with pytest.raises(ValueError):#, match='operands could not be broadcast together with shapes (9,1) (10,1) '):
            actual = ndt.maxdist(x1, y1, x2, y2)


class TestClass_quadct:
    def test_quadct_general(self):
        np.random.seed(100)
        x1 = np.random.random(10)
        y1 = np.random.random(10)
        actual = ndt.quadct(x1[0], y1[0], x1, y1)
        expected = (0.5, 0.1, 0.4, 0.0)
        assert actual == expected

    def test_quadct_simple(self):
        x1 = np.array([0, 0, 1, 2, 2])
        y1 = np.array([0, 2, 1, 0, 2])
        actual = ndt.quadct(x1[2], y1[2], x1, y1)
        expected = (0.4, 0.2, 0.2, 0.2)
        assert actual == pytest.approx(expected)

    def test_quadct_medium(self):
        x1 = np.array([0, 0, 0, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 2, 2, 2])
        y1 = np.array([0, 1, 2, 0.5, 1.5, 0, 1, 2, 0.5, 1.5, 0, 1, 2])
        actual = ndt.quadct(x1[6], y1[6], x1, y1)
        expected = (0.3846, 0.2308, 0.2308, 0.1538)
        assert actual == pytest.approx(expected, abs=1e-4)

    def test_quadct_equal_zeros(self):
        x1 = np.zeros((10))
        y1 = x1.copy()
        actual = ndt.quadct(x1[0], y1[0], x1, y1)
        expected = (1, 0, 0, 0)
        assert actual == expected

    def test_quadct_equal_ones(self):
        x1 = np.ones((10))
        y1 = x1.copy()
        actual = ndt.quadct(x1[0], y1[0], x1, y1)
        expected = (1, 0, 0, 0)
        assert actual == expected
