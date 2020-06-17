import sqlite3
from unittest import TestCase
from attribute_management import attribute_generator_publisher


class Test(TestCase):
    def test_attribute_generator_publisher(self):
        try:
            attribute_generator_publisher()
            self.assertTrue(True)
        except sqlite3.Error:
            self.fail()
