import os
import boto3
import io
from botocore.exceptions import NoCredentialsError
import json
from abc import ABC, abstractmethod
from typing import List


class FileSystem(ABC):
    subclasses = {}

    def __init__(self, base_path: str):
        self.base_path = base_path

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            "subclass_name": self.__class__.__name__,
            "base_path": self.base_path
        }

    @classmethod
    def from_dict(cls, config):
        subclass_name = config.pop(
            "subclass_name", None
        )  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def create_directory(self, kb_id: str, doc_id: str) -> None:
        pass

    @abstractmethod
    def delete_directory(self, kb_id: str, doc_id: str) -> None:
        pass

    @abstractmethod
    def save_json(self, kb_id: str, doc_id: str, file_name: str, file: dict) -> None:
        pass

    @abstractmethod
    def save_image(self, kb_id: str, doc_id: str, file_name: str, file: any) -> None:
        pass

    @abstractmethod
    def get_files(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
        pass



class LocalFileSystem(FileSystem):

    def __init__(self, base_path: str):
        super().__init__(base_path)

    def create_directory(self, kb_id: str, doc_id: str) -> None:
        """
        Create a directory to store the images of the pages
        """
        page_images_path = os.path.join(self.base_path, kb_id, doc_id)
        if os.path.exists(page_images_path):
            for file in os.listdir(page_images_path):
                os.remove(os.path.join(page_images_path, file))
            os.rmdir(page_images_path)

        # Create the folder
        os.makedirs(page_images_path, exist_ok=False)

    def delete_directory(self, kb_id: str, doc_id: str) -> None:
        """
        Delete the directory
        """
        page_images_path = os.path.join(self.base_path, kb_id, doc_id)
        if os.path.exists(page_images_path):
            for file in os.listdir(page_images_path):
                os.remove(os.path.join(page_images_path, file))
            os.rmdir(page_images_path)

    def save_json(self, kb_id: str, doc_id: str, file_name: str, file: dict) -> None:
        """
        Save the file to the local system
        """

        file_path = os.path.join(self.base_path, kb_id, doc_id, file_name)
        with open(file_path, "w") as f:
            json.dump(file, f, indent=2)
        
    def save_image(self, kb_id: str, doc_id: str, file_name: str, image: any) -> None:
        """
        Save the image to the local system
        """
        image_path = os.path.join(self.base_path, kb_id, doc_id, file_name)
        image.save(image_path, 'PNG')

    def get_files(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
        """
        Get the file from the local system
        - page_start: int - the starting page number
        - page_end: int - the ending page number (inclusive)
        """
        if page_start is None or page_end is None:
            return []
        page_images_path = os.path.join(self.base_path, kb_id, doc_id)
        image_file_paths = []
        for i in range(page_start, page_end + 1):
            image_file_path = os.path.join(page_images_path, f'page_{i}.png')
            # Make sure the file exists
            if not os.path.exists(image_file_path):
                continue
            image_file_paths.append(image_file_path)
        return image_file_paths


class S3FileSystem(FileSystem):

    def __init__(self, base_path: str, bucket_name: str, region_name: str, access_key: str, secret_key: str):
        super().__init__(base_path)
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.access_key = access_key
        self.secret_key = secret_key

    def create_s3_client(self):
        return boto3.client(
            service_name='s3',
            region_name=self.region_name,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

    def create_directory(self, kb_id: str, doc_id: str) -> None:
        """
        This function is not needed for S3
        """
        pass

    def delete_directory(self, kb_id: str, doc_id: str) -> List[dict]:
        """
        Delete the directory in S3
        """
        
        s3_client = self.create_s3_client()
        prefix = f"{kb_id}/{doc_id}/"

        # List all objects with the specified prefix
        response = s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

        # Check if there are any objects to delete
        if 'Contents' in response:
            # Prepare a list of object keys to delete
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]

            # Delete the objects
            s3_client.delete_objects(Bucket=self.bucket_name, Delete={'Objects': objects_to_delete})
            print(f"Deleted all objects in {prefix} from {self.bucket_name}.")
        else:
            print(f"No objects found in {prefix}.")
            objects_to_delete = []

        return objects_to_delete


    def save_json(self, kb_id: str, doc_id: str, file_name: str, file: dict) -> None:
        """
        Save the JSON file to S3
        """

        file_name = f"{kb_id}/{doc_id}/{file_name}"
        json_data = json.dumps(file, indent=2)  # Serialize the JSON data

        s3_client = self.create_s3_client()
        try:
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=json_data,
                ContentType='application/json'
            )
            print(f"JSON data uploaded to {self.bucket_name}/{file_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to upload JSON to S3.") from e

    def save_image(self, kb_id: str, doc_id: str, file_name: str, file: any) -> None:
        """
        Upload the file to S3
        """

        file_name = f"{kb_id}/{doc_id}/{file_name}"
        buffer = io.BytesIO()
        file.save(buffer, format='PNG')
        buffer.seek(0)  # Rewind the buffer to the beginning

        s3_client = self.create_s3_client()
        try:
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=buffer,
                ContentType='image/png'
            )
            print(f"JSON data uploaded to {self.bucket_name}/{file_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to upload image to S3.") from e


    def get_files(self, kb_id: str, doc_id: str, page_start: int, page_end: int) -> List[str]:
        """
        Get the file from S3
        - page_start: int - the starting page number
        - page_end: int - the ending page number (inclusive)
        """
        if page_start is None or page_end is None:
            return []
        filenames = [f"{kb_id}/{doc_id}/page_{i}.png" for i in range(page_start, page_end + 1)]
        s3_client = self.create_s3_client()
        file_paths = []
        for filename in filenames:
            output_folder = os.path.join(self.base_path, kb_id, doc_id)
            if not os.path.exists(output_folder):
                try:
                    os.makedirs(output_folder)
                except FileExistsError:
                    # Since this function can be called in parallel, the folder may have been created by another process
                    pass
            output_filepath = os.path.join(self.base_path, filename)
            print ("filename", filename)
            try:
                s3_client.download_file(
                    self.bucket_name,
                    filename,
                    output_filepath
                )
                file_paths.append(output_filepath)
                print ("File downloaded successfully.")
            except Exception as e:
                print ("Error downloading file:", e)
            
        return file_paths
    

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            "bucket_name": self.bucket_name,
            "region_name": self.region_name,
            "access_key": self.access_key,
            "access_secret": self.secret_key
        })
        return base_dict