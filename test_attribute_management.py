import sqlite3
from unittest import TestCase
from attr_generators.attribute_management import attribute_generator_publisher, get_active_attr_generators


class Test(TestCase):
    def test_attribute_generator_publisher(self):
        try:
            attribute_generator_publisher()
            self.assertTrue(True)
        except sqlite3.Error:
            self.fail()


class Test(TestCase):
    def test_get_active_attr_generators(self):
        attribute_generator_publisher()
        conn = sqlite3.Connection('example.db')

        c = conn.cursor()

        c.execute("UPDATE MATTR SET active = ?", (True,))

        c.execute("SELECT * FROM MATTR")

        attributes = len(c.fetchall())

        activeones = get_active_attr_generators(conn)

        if len(activeones) == attributes:
            print(activeones)
            self.assertTrue(True)
        else:
            self.fail()
        conn.rollback()
        conn.close()
