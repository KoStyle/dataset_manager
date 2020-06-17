import sqlite3
from unittest import TestCase

from user_case import UserCase

from MDT import load_usercase_set_IMBD


class TestUserCase(TestCase):
    def test_db_log_user(self):
        load_usercase_set_IMBD()
        user_cases = load_usercase_set_IMBD()
        uc: UserCase = list(user_cases.items())[0][1]

        uc.user_id = "TEST_ID"

        conn = sqlite3.Connection('example.db')

        conn.cursor().execute("DELETE FROM MUSR WHERE uid='TEST_ID'")
        conn.cursor().execute("DELETE FROM MREVS WHERE uid='TEST_ID'")

        uc.db_log_user(conn)

        c=conn.cursor()
        c.execute("SELECT * from MUSR WHERE uid='TEST_ID'")
        self.assertTrue(c.fetchone() is not None, "Mensaje?, para bien?")

        conn.cursor().execute("DELETE FROM MUSR WHERE uid='TEST_ID'")
        conn.cursor().execute("DELETE FROM MREVS WHERE uid='TEST_ID'")
        c.close()
        conn.commit()
        conn.close()

