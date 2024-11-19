import os
import sys
import unittest
from pdf2image import convert_from_path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from dsparse.file_parsing.file_system import LocalFileSystem, S3FileSystem


class TestLocalFileSystem(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.kb_id = "test_kb"
        self.doc_id = "test_doc"
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dsparse_file_system_test'))
        self.file_system = LocalFileSystem(base_path=self.base_path)

    def test__001_create_directory(self):

        self.file_system.create_directory(self.kb_id, self.doc_id)
        # Check if the directory was created
        self.assertTrue(os.path.exists(os.path.join(self.base_path, self.kb_id, self.doc_id)))

    def test__002_save_json(self):

        test_json = {
            "test_key": "test_value",
            "test_key_2": "test_value_2"
        }
        file_name = "elements.json"
        self.file_system.save_json(self.kb_id, self.doc_id, file_name, test_json)
        # Make sure the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.base_path, self.kb_id, self.doc_id, file_name)))

    def test__003_save_image(self):

        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../tests/data/mck_energy_first_5_pages.pdf'))
        images = convert_from_path(pdf_path, dpi=150)
        
        file_name = "page_0.png"
        self.file_system.save_image(self.kb_id, self.doc_id, file_name, images[0])
        # Make sure the file was saved
        self.assertTrue(os.path.exists(os.path.join(self.base_path, self.kb_id, self.doc_id, file_name)))

        file_name = "page_1.png"
        self.file_system.save_image(self.kb_id, self.doc_id, file_name, images[1])

    def test__004_get_files(self):

        files = self.file_system.get_files(self.kb_id, self.doc_id, page_start=0, page_end=1)
        self.assertTrue(len(files) == 2)

        # Test for a page that doesn't exist
        files = self.file_system.get_files(self.kb_id, self.doc_id, page_start=2, page_end=2)
        self.assertTrue(len(files) == 0)

    def test__005_get_all_files(self):
            
        files = self.file_system.get_all_png_files(self.kb_id, self.doc_id)
        self.assertTrue(len(files) == 2)
        # The only file returned should be page_0.png
        self.assertTrue(files[0] == os.path.join(self.base_path, self.kb_id, self.doc_id, "page_0.png"))
        self.assertTrue(files[1] == os.path.join(self.base_path, self.kb_id, self.doc_id, "page_1.png"))

    def test__006_delete_directory(self):

        self.file_system.delete_directory(self.kb_id, self.doc_id)
        # Check if the directory was deleted
        self.assertFalse(os.path.exists(os.path.join(self.base_path, self.kb_id, self.doc_id)))

    def test__007_delete_kb(self):
        self.file_system.create_directory(self.kb_id, self.doc_id)

        # Add a file to the directory
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../tests/data/mck_energy_first_5_pages.pdf'))
        images = convert_from_path(pdf_path, dpi=150)
        
        file_name = "page_0.png"
        self.file_system.save_image(self.kb_id, self.doc_id, file_name, images[0])
        self.file_system.save_image(self.kb_id, "", file_name, images[0])

        self.file_system.delete_kb(self.kb_id)
        # Check if the directory was deleted
        self.assertFalse(os.path.exists(os.path.join(self.base_path, self.kb_id)))

    @classmethod
    def tearDownClass(self):
        try:
            os.system(f"rm -rf {self.base_path}")
        except:
            pass


class TestS3FileSystem(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.kb_id = "test_kb"
        self.doc_id = "test_doc"
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dsparse_file_system_test'))
        self.s3_file_system = S3FileSystem(
            base_path=self.base_path,
            bucket_name=os.environ["AWS_S3_BUCKET_NAME"],
            region_name=os.environ["AWS_S3_REGION"],
            access_key=os.environ["AWS_S3_ACCESS_KEY"],
            secret_key=os.environ["AWS_S3_SECRET_KEY"]
        )

    def test__001_create_directory(self):

        self.s3_file_system.create_directory(self.kb_id, self.doc_id)
        # Nothing actually happens, but it shouldn't cause any errors

    def test__002_save_json(self):

        test_json = {
            "test_key": "test_value",
            "test_key_2": "test_value_2"
        }
        file_name = "elements.json"
        self.s3_file_system.save_json(self.kb_id, self.doc_id, file_name, test_json)

    def test__003_save_image(self):

        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../tests/data/mck_energy_first_5_pages.pdf'))
        images = convert_from_path(pdf_path, dpi=150)
        
        file_name = "page_0.png"
        self.s3_file_system.save_image(self.kb_id, self.doc_id, file_name, images[0])
        file_name = "page_1.png"
        self.s3_file_system.save_image(self.kb_id, self.doc_id, file_name, images[1])

    def test__004_get_files(self):

        files = self.s3_file_system.get_files(self.kb_id, self.doc_id, page_start=0, page_end=1)
        self.assertTrue(len(files) == 2)
        # Make sure the file was saved locally to the base path
        self.assertTrue(os.path.exists(os.path.join(self.base_path, self.kb_id, self.doc_id, "page_0.png")))

        # Try it again with the file already saved locally (Shouldn't cause any issues)
        files = self.s3_file_system.get_files(self.kb_id, self.doc_id, page_start=0, page_end=1)
        self.assertTrue(len(files) == 2)

        # Test for a page that doesn't exist
        files = self.s3_file_system.get_files(self.kb_id, self.doc_id, page_start=2, page_end=2)
        self.assertTrue(len(files) == 0)

    def test__005_get_all_files(self):

        files = self.s3_file_system.get_all_png_files(self.kb_id, self.doc_id)
        self.assertTrue(len(files) == 2)
        # The only files returned should be page_0.png and page_1.png
        self.assertTrue(files[0] == os.path.join(self.base_path, self.kb_id, self.doc_id, "page_0.png"))
        self.assertTrue(files[1] == os.path.join(self.base_path, self.kb_id, self.doc_id, "page_1.png"))

    def test__006_delete_directory(self):

        objects_deleted = self.s3_file_system.delete_directory(self.kb_id, self.doc_id)
        self.assertTrue(len(objects_deleted) == 3)
        self.assertTrue(objects_deleted[0]["Key"] == f"{self.kb_id}/{self.doc_id}/elements.json")
        self.assertTrue(objects_deleted[1]["Key"] == f"{self.kb_id}/{self.doc_id}/page_0.png")

        # Try to delete a directory that doesn't exist
        objects_deleted = self.s3_file_system.delete_directory(self.kb_id, self.doc_id)
        self.assertTrue(len(objects_deleted) == 0)

    def test__007_delete_kb(self):
        self.s3_file_system.create_directory(self.kb_id, self.doc_id)

        # Add a file to the directory
        pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../tests/data/mck_energy_first_5_pages.pdf'))
        images = convert_from_path(pdf_path, dpi=150)
        
        file_name = "page_0.png"
        self.s3_file_system.save_image(self.kb_id, self.doc_id, file_name, images[0])

        objects_deleted = self.s3_file_system.delete_kb(self.kb_id)
        self.assertTrue(len(objects_deleted) == 1)
        self.assertTrue(objects_deleted[0]["Key"] == f"{self.kb_id}/{self.doc_id}/page_0.png")
        

    @classmethod
    def tearDownClass(self):
        try:
            os.system(f"rm -rf {self.base_path}")
        except:
            pass


if __name__ == '__main__':
    unittest.main()