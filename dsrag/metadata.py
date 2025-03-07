# Metadata storage handling
import os
from decimal import Decimal
import json
from typing import Any
from abc import ABC, abstractmethod
from dsrag.utils.imports import boto3

class MetadataStorage(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self) -> dict:
        pass

    @abstractmethod
    def save(self, kb: dict) -> None:
        pass

    @abstractmethod
    def delete(self) -> None:
        pass

class LocalMetadataStorage(MetadataStorage):

    def __init__(self, storage_directory: str) -> None:
        super().__init__()
        self.storage_directory = storage_directory

    def get_metadata_path(self, kb_id: str) -> str:
        return os.path.join(self.storage_directory, "metadata", f"{kb_id}.json")

    def kb_exists(self, kb_id: str) -> bool:
        metadata_path = self.get_metadata_path(kb_id)
        return os.path.exists(metadata_path)
    
    def load(self, kb_id: str) -> dict:
        metadata_path = self.get_metadata_path(kb_id)
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return data

    def save(self, full_data: dict, kb_id: str):
        metadata_path = self.get_metadata_path(kb_id)
        metadata_dir = os.path.join(self.storage_directory, "metadata")
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)

        with open(metadata_path, "w") as f:
            json.dump(full_data, f, indent=4)

    def delete(self, kb_id: str):
        metadata_path = self.get_metadata_path(kb_id)
        os.remove(metadata_path)



def convert_numbers_to_decimal(obj: Any) -> Any:
    """
    Recursively traverse the object, converting all integers and floats to Decimals
    """
    if isinstance(obj, dict):
        return {k: convert_numbers_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numbers_to_decimal(item) for item in obj]
    elif isinstance(obj, bool):
        return obj  # Leave booleans as is
    elif isinstance(obj, (int, float)):
        return Decimal(str(obj))
    else:
        return obj
    

def convert_decimal_to_numbers(obj: Any) -> Any:
    """
    Recursively traverse the object, converting all Decimals to integers or floats
    """
    if isinstance(obj, dict):
        return {k: convert_decimal_to_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_numbers(item) for item in obj]
    elif isinstance(obj, Decimal):
        if obj == Decimal('1') or obj == Decimal('0'):
            # Convert to bool
            return bool(obj)
        elif obj % 1 == 0:
            # Convert to int
            return int(obj)
        else:
            # Convert to float
            return float(obj)
    else:
        return obj


class DynamoDBMetadataStorage(MetadataStorage):

    def __init__(self, table_name: str, region_name: str, access_key: str, secret_key: str) -> None:
        self.table_name = table_name
        self.region_name = region_name
        self.access_key = access_key
        self.secret_key = secret_key

    def create_dynamo_client(self):
        dynamodb_client = boto3.resource(
            'dynamodb',
            region_name=self.region_name,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        return dynamodb_client
    
    def create_table(self):
        dynamodb_client = self.create_dynamo_client()
        try:
            dynamodb_client.create_table(
                TableName="metadata_storage",
                KeySchema=[
                    {
                        'AttributeName': 'kb_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'kb_id',
                        'AttributeType': 'S'  # 'S' for String
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
        except Exception as e:
            # Probably the table already exists
            print (e)

    def kb_exists(self, kb_id: str) -> bool:
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        response = table.get_item(Key={'kb_id': kb_id})
        return 'Item' in response
    
    def load(self, kb_id: str) -> dict:
        # Read the data from the dynamo table
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        response = table.get_item(Key={'kb_id': kb_id})
        data = response.get('Item', {}).get('metadata', {})
        kb_metadata = {
            key: value for key, value in data.items() if key != "components"
        }
        components = data.get("components", {})
        full_data = {**kb_metadata, "components": components}
        converted_data = convert_decimal_to_numbers(full_data)
        return converted_data

    def save(self, full_data: dict, kb_id: str) -> None:

        # Check if any of the items are a float or int, and convert them to Decimal
        converted_data = convert_numbers_to_decimal(full_data)

        # Upload this data to the dynamo table, where the kb_id is the primary key
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        table.put_item(Item={'kb_id': kb_id, 'metadata': converted_data})

    def delete(self, kb_id: str) -> None:
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        table.delete_item(Key={'kb_id': kb_id})

