import sqlite3
from unittest import TestCase

from io_management import load_dataset_files_IMBD
from user_case import UserCase


class TestUserCase(TestCase):
    def test_db_log_user(self):
        user_cases = load_dataset_files_IMBD()
        uc: UserCase = list(user_cases.items())[0][1]

        uc.user_id = "TEST_ID"
        rev_amount = len(uc.reviews)

        conn = sqlite3.Connection('example.db')

        conn.cursor().execute("DELETE FROM MUSR WHERE uid='TEST_ID'")
        conn.cursor().execute("DELETE FROM MREVS WHERE uid='TEST_ID'")

        self.assertTrue(uc.db_log_user(conn))

        c = conn.cursor()
        c.execute("SELECT * from MUSR WHERE uid='TEST_ID'")
        self.assertTrue(c.fetchone() is not None, "Mensaje?, para bien?")
        c.execute("SELECT * FROM MREVS WHERE uid= 'TEST_ID'")
        self.assertTrue(len(c.fetchall()) == 1000)

        uc.reviews = {}
        uc.db_load_reviews(conn)
        self.assertTrue(len(uc.reviews) == rev_amount)  #We check we load the same amount of reviews

        conn.cursor().execute("DELETE FROM MUSR WHERE uid='TEST_ID'")
        conn.cursor().execute("DELETE FROM MREVS WHERE uid='TEST_ID'")
        c.close()
        conn.commit()
        conn.close()
