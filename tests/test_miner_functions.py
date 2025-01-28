import pytest
from folding.miners.folding_miner import check_sqlite_table
import unittest
from unittest.mock import patch, MagicMock
import sqlite3


class TestCheckSqliteTable(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            (
                1,
                "job1",
                "pdb1",
                "2025-01-01 00:00:00",
                10,
                True,
                '{"pdb": "link_to_pdb"}',
                '{"ff": "file.xml"}',
            ),
            (
                2,
                "job2",
                "pdb2",
                "2025-01-02 00:00:00",
                5,
                False,
                '{"pdb": "link_to_pdb2"}',
                '{"ff": "file2.xml"}',
            ),
        ]
        self.columns = [
            "id",
            "job_id",
            "pdb_id",
            "created_at",
            "priority",
            "is_organic",
            "s3_links",
            "system_config",
        ]

    @patch("folding.utils.logger.logger")
    @patch("sqlite3.connect")
    def test_check_sqlite_table_success(self, mock_connect, mock_logger):
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = self.sample_data
        mock_cursor.description = [(column,) for column in self.columns]
        mock_connect.return_value.__enter__.return_value.cursor.return_value = (
            mock_cursor
        )

        result = check_sqlite_table(db_path="test_path.db", max_workers=2)
        print(len(result))
        # check if result is correctly formatted
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(result[key], dict) for key in result))

        # check if all expected keys are in the results
        expected_keys = [
            "job_id",
            "pdb_id",
            "created_at",
            "priority",
            "is_organic",
            "s3_links",
            "system_config",
        ]
        for job_details in result.values():
            self.assertTrue(all(key in job_details for key in expected_keys))

    @patch("folding.utils.logger.logger")
    @patch("sqlite3.connect")
    def test_check_sqlite_table_error(self, mock_connect, mock_logger):
        mock_connect.side_effect = sqlite3.Error("Fake SQL Error")

        result = check_sqlite_table(db_path="test_path.db", max_workers=2)

        # check if result is None
        self.assertIsNone(result)
