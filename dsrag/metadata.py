# Metadata storage handling
import os
import sys
from decimal import Decimal
import boto3
import json
from typing import Any
from abc import ABC, abstractmethod

class MetadataStorage(ABC):

    def __init__(self, kb_id: str) -> None:
        self.kb_id = kb_id

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

    def __init__(self, kb_id: str, storage_directory: str) -> None:
        super().__init__(kb_id)
        self.storage_directory = storage_directory
        self.metadata_path = os.path.join(self.storage_directory, "metadata", f"{self.kb_id}.json")
    
    def load(self) -> dict:
        with open(self.metadata_path, "r") as f:
            data = json.load(f)
            self.kb_metadata = {
                key: value for key, value in data.items() if key != "components"
            }
            components = data.get("components", {})
        return components

    def save(self, full_data: dict):

        metadata_dir = os.path.join(self.storage_directory, "metadata")
        if not os.path.exists(metadata_dir):
            os.makedirs(metadata_dir)

        with open(self.metadata_path, "w") as f:
            json.dump(full_data, f, indent=4)

    def delete(self):
        os.remove(self.metadata_path)



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

    def __init__(self, kb_id: str, table_name: str) -> None:
        super().__init__(kb_id)
        self.table_name = table_name

    def create_dynamo_client(self):
        dynamodb_client = boto3.resource(
            'dynamodb',
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_DYNAMO_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("AWS_DYNAMO_SECRET_KEY")
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
    
    def load(self) -> dict:
        # Read the data from the dynamo table
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        response = table.get_item(Key={'kb_id': self.kb_id})
        data = response.get('Item', {}).get('metadata', {})
        kb_metadata = {
            key: value for key, value in data.items() if key != "components"
        }
        components = data.get("components", {})
        full_data = {**kb_metadata, "components": components}
        converted_data = convert_decimal_to_numbers(full_data)
        return converted_data

    def save(self, full_data: dict) -> None:

        # Check if any of the items are a float or int, and convert them to Decimal
        converted_data = convert_numbers_to_decimal(full_data)

        # Upload this data to the dynamo table, where the kb_id is the primary key
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        table.put_item(Item={'kb_id': self.kb_id, 'metadata': converted_data})

    def delete(self):
        dynamodb_client = self.create_dynamo_client()
        table = dynamodb_client.Table(self.table_name)
        table.delete_item(Key={'kb_id': self.kb_id})

