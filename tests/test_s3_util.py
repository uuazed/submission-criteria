
from submission_criteria import s3_util

# System
import unittest
from unittest.mock import MagicMock
import zipfile
import os
import tempfile


def create_zipfile(s3_path, bucket, local_path):
    zf = zipfile.ZipFile(local_path, "w")
    zf.writestr("test.txt", "")
    zf.close()


def create_file(s3_path, bucket, local_path):
    with open(local_path, "w") as filehandler:
        filehandler.write("")


class TestS3Util(unittest.TestCase):

    def test_download_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            round_number = 99
            file_manager = s3_util.FileManager(temp_dir)

            # mock actual download and create zip file as `side_effect`
            file_manager.s3.meta.client.download_file = MagicMock()
            file_manager.s3.meta.client.download_file.side_effect = create_zipfile

            # call function
            extract_dir = file_manager.download_dataset(round_number)

            # check if the expected file exists
            target = os.path.join(extract_dir, "test.txt")
            self.assertTrue(os.path.exists(target))

    def test_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = s3_util.FileManager(temp_dir)

            # mock actual download and create file as `side_effect`
            file_manager.s3.meta.client.download_file = MagicMock()
            file_manager.s3.meta.client.download_file.side_effect = create_file

            # call function
            paths = file_manager.download(["a.csv", "b.csv"])

            # check if the expected files exist
            for path in paths:
                self.assertTrue(os.path.exists(path))
                self.assertTrue(os.path.exists(path))
            # check if it triggered two downloads
            count1 = file_manager.s3.meta.client.download_file.call_count
            self.assertEqual(count1, 2)

            # getting the same file again should not trigger a download
            paths = file_manager.download(["a.csv"])
            count2 = file_manager.s3.meta.client.download_file.call_count
            self.assertEqual(count2, 2)

    def test___hash__(self):
        hash_val = hash(s3_util.FileManager("."))
        self.assertEqual(hash_val, 90210)


if __name__ == '__main__':
    unittest.main()
