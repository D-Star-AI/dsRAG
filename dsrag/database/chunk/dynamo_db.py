import os
from typing import Any, Optional
from decimal import Decimal
import time
from dsrag.utils.imports import boto3
from dsrag.database.chunk.db import ChunkDB
from dsrag.database.chunk.types import FormattedDocument


def get_key():
    """Helper function to get the Key class from boto3.dynamodb.conditions"""
    return boto3.dynamodb.conditions.Key


def process_items(items):

    def convert_decimal(obj):
        if isinstance(obj, list):
            return [convert_decimal(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        else:
            return obj

    return convert_decimal(items)


class DynamoDB(ChunkDB):

    def __init__(self, kb_id: str, table_name: str = None, billing_mode: str = "PAY_PER_REQUEST") -> None:
        self.kb_id = kb_id
        self.billing_mode = billing_mode
        if table_name is not None:
            self.table_name = table_name
        else:
            # Strip the kb of any spaces
            kb_id = kb_id.replace(" ", "_")
            self.table_name = f"{kb_id}_chunks"

        self.columns = [
            'doc_id',
            'chunk_index',
            'created_on',
            'document_title',
            'document_summary',
            'section_title',
            'section_summary',
            'supp_id',
            'chunk_text',
            'chunk_length',
            'chunk_page_start',
            'chunk_page_end',
            'is_visual',
            'metadata'
        ]

        """
        - **Partition Key (`doc_id`):** String
        - **Sort Key (`chunk_index`):** Number
        - **Additional Attributes:**
        - `supp_id` (String)
        - `document_title` (String)
        - `document_summary` (String)
        - `section_title` (String)
        - `section_summary` (String)
        - `chunk_text` (String)
        - `chunk_length` (Number)
        - `chunk_page_start` (Number)
        - `chunk_page_end` (Number)
        - `is_visual` (Boolean)
        - `created_on` (String) or (Number) depending on your timestamp format
        - `metadata` (String) or (Map)
        """

        # If the table name is not provided, create a new table
        # The name will be the kb_id_ + "chunks"
        if table_name is None:
            print ("Creating table")
            table_name = f"{kb_id}_chunks"
            self.create_db_table(table_name)

        # If the table name is provided, check if the table exists
        # If the table does not exist, create a new table

    def check_table_status(self):
        # Need to make sure the table has been created before proceeding
        dynamodb = self.create_dynamo_client()
        table = dynamodb.Table(self.table_name)
        try:
            response = table.table_status
            return response
        except Exception as e:
            print(e)
            return None
    
    def create_dynamo_client(self):
        dynamodb_client = boto3.resource(
            'dynamodb',
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("AWS_DYNAMO_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("AWS_DYNAMO_SECRET_KEY")
        )
        return dynamodb_client


    def create_db_table(self, table_name: str) -> None:
        dynamodb = self.create_dynamo_client()

        try:
            response = dynamodb.create_table(
                TableName=table_name,
                AttributeDefinitions=[
                    {'AttributeName': 'doc_id', 'AttributeType': 'S'},
                    {'AttributeName': 'chunk_index', 'AttributeType': 'N'},
                    {'AttributeName': 'supp_id', 'AttributeType': 'S'},
                ],
                KeySchema=[
                    {'AttributeName': 'doc_id', 'KeyType': 'HASH'},       # Partition Key
                    {'AttributeName': 'chunk_index', 'KeyType': 'RANGE'},  # Sort Key
                ],
                GlobalSecondaryIndexes=[
                    {
                        'IndexName': 'SuppIdIndex',
                        'KeySchema': [
                            {'AttributeName': 'supp_id', 'KeyType': 'HASH'},   # Partition Key
                            {'AttributeName': 'doc_id', 'KeyType': 'RANGE'},   # Sort Key
                        ],
                        'Projection': {
                            'ProjectionType': 'ALL',  # Include all attributes (adjust as needed)
                        },
                    },
                ],
                BillingMode=self.billing_mode,  # On-demand billing
            )
            print("Table creation initiated. Status:", response)
            time.sleep(5)
        except Exception as e:
            print(e)


    def add_document(self, doc_id: str, chunks: dict[int, dict[str, Any]], supp_id: str = "", metadata: dict = {}) -> None:
        # Initialize DynamoDB resource
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)

        table_status = self.check_table_status()
        if table_status is None:
            print ("Table not created")
            return
        elif table_status == "CREATING":
            print ("Table is still creating")
            time.sleep(10)
            table_status = self.check_table_status()
            if table_status == "CREATING":
                print ("Table is still creating")
                time.sleep(10)

        # Create a 'created_on' timestamp
        created_on = int(time.time())

        with table.batch_writer() as batch:
            for chunk_index, chunk in chunks.items():
                try:
                    # Initialize the item with mandatory attributes
                    item = {
                        'doc_id': doc_id,
                        'chunk_index': Decimal(str(chunk_index)),
                        'created_on': Decimal(str(created_on))
                    }

                    if supp_id:
                        item['supp_id'] = supp_id

                    if metadata:
                        item['metadata'] = metadata  # Assuming metadata is a dict

                    # Process 'chunk_text'
                    chunk_text = chunk.get('chunk_text')
                    if chunk_text and chunk_text.strip() != '':
                        item['chunk_text'] = chunk_text
                        item['chunk_length'] = Decimal(str(len(chunk_text)))

                    # Process other string attributes, avoiding empty strings
                    for attr in ['document_title', 'document_summary', 'section_title', 'section_summary']:
                        value = chunk.get(attr)
                        if value and value.strip() != '':
                            item[attr] = value

                    # Process numerical attributes 'chunk_page_start', 'chunk_page_end'
                    for attr in ['chunk_page_start', 'chunk_page_end']:
                        value = chunk.get(attr)
                        if value is not None:
                            item[attr] = Decimal(str(value))

                    # Process boolean attributes
                    if 'is_visual' in chunk:
                        is_visual = chunk['is_visual']
                        if isinstance(is_visual, bool):
                            item['is_visual'] = is_visual
                        else:
                            item['is_visual'] = bool(is_visual)

                    # Remove attributes with None values or empty strings
                    item = {k: v for k, v in item.items() if v not in [None, '', []]}

                    # Write the item to DynamoDB
                    batch.put_item(Item=item)

                except Exception as e:
                    print(f"Error processing chunk_index {chunk_index}: {e}")
                    # Handle exceptions as needed (e.g., log, skip, or raise)


    def remove_document(self, doc_id: str) -> None:
        # Have to get all the items first
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        
        # Get all items from the table with the given doc_id
        response = table.query(
            KeyConditionExpression=get_key()('doc_id').eq(doc_id),
            ProjectionExpression='doc_id, chunk_index'
        )
        items = response.get('Items', [])

        # Delete the items in batches (25 items per batch)
        # Keep track of which items have been deleted
        deleted_items = []
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(
                    Key={
                        'doc_id': item['doc_id'],
                        'chunk_index': item['chunk_index']
                    }
                )
                deleted_items.append(item['doc_id'])

        return deleted_items


    def get_document(self, doc_id: str, include_content: bool = False) -> Optional[FormattedDocument]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)

        # Define the attributes to retrieve
        projection_attributes = ['supp_id', 'document_title', 'document_summary', 'created_on', 'metadata']
        if include_content:
            projection_attributes += ['chunk_text', 'chunk_index']

        # Build the ProjectionExpression and ExpressionAttributeNames to handle reserved words
        expression_attribute_names = {f'#{attr}': attr for attr in projection_attributes}
        projection_expression = ', '.join(expression_attribute_names.keys())

        try:
            # Query the table for all items with the given doc_id
            response = table.query(
                KeyConditionExpression=get_key()('doc_id').eq(doc_id),
                ProjectionExpression=projection_expression,
                ExpressionAttributeNames=expression_attribute_names
            )

            items = response.get('Items', [])

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = table.query(
                    KeyConditionExpression=get_key()('doc_id').eq(doc_id),
                    ProjectionExpression=projection_expression,
                    ExpressionAttributeNames=expression_attribute_names,
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))

            # If no items found, return None
            if not items:
                return None

            # Initialize variables
            full_document_string = ""
            chunks = []

            # Process items
            for item in items:
                # Extract attributes from the first item
                if items.index(item) == 0:
                    supp_id = item.get('supp_id')
                    title = item.get('document_title')
                    summary = item.get('document_summary')
                    created_on = item.get('created_on')
                    if isinstance(created_on, Decimal):
                        created_on = int(created_on)
                    metadata = item.get('metadata')

                    # If metadata is stored as a string, convert it back to a dictionary
                    if metadata and isinstance(metadata, str):
                        metadata = eval(metadata)  # Be cautious with eval; consider safer alternatives

                # Collect chunks if include_content is True
                if include_content:
                    chunk_text = item.get('chunk_text', '')
                    chunk_index = item.get('chunk_index')
                    if isinstance(chunk_index, Decimal):
                        chunk_index = int(chunk_index)
                    chunks.append((chunk_index, chunk_text))

            # Concatenate chunks
            if include_content and chunks:
                # Sort the chunks based on chunk_index
                chunks.sort(key=lambda x: x[0])
                # Join chunk_texts with newline character
                full_document_string = '\n'.join(chunk_text for _, chunk_text in chunks)

            return FormattedDocument(
                id=doc_id,
                supp_id=supp_id,
                title=title,
                content=full_document_string if include_content else None,
                summary=summary,
                created_on=created_on,
                metadata=metadata,
                chunk_count=len(items)
            )

        except Exception as e:
            print(f"Error retrieving document '{doc_id}': {e}")
            # Handle exceptions as needed
            return None

    def get_chunk_text(self, doc_id: str, chunk_index: int) -> Optional[str]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)

        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='chunk_text'
        )

        item = response.get('Item')
        if item:
            return item.get('chunk_text')
        else:
            return None

    def get_is_visual(self, doc_id: str, chunk_index: int) -> Optional[bool]:
        # Get the 'is_visual' attribute for the given doc_id and chunk_index
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='is_visual'
        )
        item = response.get('Item')
        if item:
            return item.get('is_visual')
        else:
            return None

    def get_chunk_page_numbers(self, doc_id: str, chunk_index: int) -> Optional[tuple[int, int]]:
        # Get the chunk page start and end
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='chunk_page_start, chunk_page_end'
        )
        item = response.get('Item')
        if item:
            # Convert the Decimal values to integers
            page_start = int(item.get('chunk_page_start', 0))
            page_end = int(item.get('chunk_page_end', 0))
            return page_start, page_end
        else:
            return None, None

    def get_document_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='document_title'
        )
        item = response.get('Item')
        if item:
            return item.get('document_title')
        else:
            return None

    def get_document_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='document_summary'
        )

        item = response.get('Item')
        if item:
            return item.get('document_summary')
        else:
            return None

    def get_section_title(self, doc_id: str, chunk_index: int) -> Optional[str]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='section_title'
        )
        item = response.get('Item')
        if item:
            return item.get('section_title')
        else:
            return None

    def get_section_summary(self, doc_id: str, chunk_index: int) -> Optional[str]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.get_item(
            Key={
                'doc_id': doc_id,
                'chunk_index': chunk_index
            },
            ProjectionExpression='section_summary'
        )
        item = response.get('Item')
        if item:
            return item.get('section_summary')
        else:
            return None

    def get_all_doc_ids(self, supp_id: Optional[str] = None) -> list[str]:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)

        doc_ids = set()

        try:
            if supp_id:
                # Query the GSI 'SuppIdIndex'
                response = table.query(
                    IndexName='SuppIdIndex',
                    KeyConditionExpression=get_key()('supp_id').eq(supp_id),
                    ProjectionExpression='doc_id',
                )
            else:
                # Scan the table
                response = table.scan(
                    ProjectionExpression='doc_id',
                )

            items = response.get('Items', [])
            for item in items:
                doc_ids.add(item['doc_id'])

            # Handle pagination
            while 'LastEvaluatedKey' in response:
                if supp_id:
                    response = table.query(
                        IndexName='SuppIdIndex',
                        KeyConditionExpression=get_key()('supp_id').eq(supp_id),
                        ProjectionExpression='doc_id',
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                else:
                    response = table.scan(
                        ProjectionExpression='doc_id',
                        ExclusiveStartKey=response['LastEvaluatedKey']
                    )
                items = response.get('Items', [])
                for item in items:
                    doc_ids.add(item['doc_id'])

        except Exception as e:
            print(f"Error retrieving doc_ids: {e}")
            # Optionally, re-raise the exception or handle it accordingly
            raise

        return list(doc_ids)

    def get_document_count(self) -> int:
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)

        doc_ids = set()

        response = table.scan(
            ProjectionExpression='doc_id'
        )

        items = response.get('Items', [])
        for item in items:
            doc_ids.add(item['doc_id'])

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                ProjectionExpression='doc_id',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items = response.get('Items', [])
            for item in items:
                doc_ids.add(item['doc_id'])

        return len(doc_ids)

    def get_total_num_characters(self) -> int:
        # Not super feasible to calculate this in DynamoDB
        pass

    def delete(self) -> None:
        # Delete the dynamo db table
        dynamo_db = self.create_dynamo_client()
        table = dynamo_db.Table(self.table_name)
        response = table.delete()
        return response

    def to_dict(self) -> dict[str, str]:
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "table_name": self.table_name,
        }