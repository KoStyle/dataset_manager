import sqlite3
from unittest import TestCase

from dataset_io.io_management import load_dataset_files_IMBD, load_dataset_from_db
from dataset_io.user_case import UserCase


class TestUserCase(TestCase):
    def test_db_log_user(self):
        user_cases = load_dataset_files_IMBD()
        i = 0
        for key in user_cases:
            user_cases[key].user_id = "TEST_ID" + i.__str__()
            user_cases[key].dataset = "FAKE"
            i += 1

        uc: UserCase = list(user_cases.items())[0][1]
        rev_amount = len(uc.reviews)
        uc_amount = len(user_cases)
        conn = sqlite3.Connection('example.db')

        conn.cursor().execute("DELETE FROM MUSR WHERE uid LIKE 'TEST_ID%'")
        conn.cursor().execute("DELETE FROM MREVS WHERE uid LIKE 'TEST_ID%'")

        for key in user_cases:
            uc = user_cases[key]
            self.assertTrue(uc.db_log_user(conn))

        c = conn.cursor()
        c.execute("SELECT * from MUSR WHERE uid = ?", (uc.user_id,))
        self.assertTrue(c.fetchone() is not None, "Mensaje?, para bien?")
        c.execute("SELECT * FROM MREVS WHERE uid = ?", (uc.user_id,))
        self.assertTrue(len(c.fetchall()) == 1000)

        uc.reviews = {}
        uc.db_load_reviews(conn)
        self.assertTrue(len(uc.reviews) == rev_amount)  # We check we load the same amount of reviews

        user_cases = {}
        user_cases = load_dataset_from_db(conn, "FAKE")
        self.assertTrue(len(user_cases) == uc_amount)

        conn.cursor().execute("DELETE FROM MUSR WHERE uid LIKE 'TEST_ID%'")
        conn.cursor().execute("DELETE FROM MREVS WHERE uid LIKE 'TEST_ID%'")
        c.close()
        conn.commit()
        conn.close()
