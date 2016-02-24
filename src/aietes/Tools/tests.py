# coding=utf-8
from unittest import TestCase

__author__ = 'bolster'

import aietes.Tools as Tools
import os
import uuid


class TestGetResultsPath(TestCase):
    def test_get_results_path_not_none_or_empty(self):
        self.assertRaises(ValueError, Tools.get_results_path, None)
        self.assertRaises(TypeError, Tools.get_results_path)

    def test_get_results_path_default_not_made(self):
        proposed = str(uuid.uuid4())
        list_before = os.listdir(Tools._results_dir)
        path = Tools.get_results_path(proposed)
        list_after = os.listdir(Tools._results_dir)
        self.assertListEqual(list_before, list_after)
        self.assertIsNotNone(path)

    def test_get_results_path_default_made(self):
        proposed = str(uuid.uuid4())
        result_dir = '/dev/shm/' + str(uuid.uuid4())

        path = Tools.get_results_path(proposed, results_dir=result_dir, make=True)

        self.assertTrue(os.path.exists(result_dir), "Path: {0}".format(result_dir))
        os.rmdir(result_dir)
        self.assertFalse(os.path.exists(result_dir))
