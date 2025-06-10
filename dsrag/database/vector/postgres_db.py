from typing import Optional, Sequence
import json
import numpy as np

from dsrag.database.vector.db import VectorDB
from dsrag.database.vector.types import VectorSearchResult, MetadataFilter, ChunkMetadata, Vector
from dsrag.utils.imports import LazyLoader

# Lazy load PostgreSQL dependencies
psycopg2 = LazyLoader("psycopg2", "psycopg2-binary")
pgvector = LazyLoader("pgvector")

# We'll import register_vector when needed to avoid immediate import


def format_metadata_filter(metadata_filter: MetadataFilter) -> dict:
    """
    Format the metadata filter to be used in the ChromaDB query method.

    Args:
        metadata_filter (dict): The metadata filter.

    Returns:
        dict: The formatted metadata filter.
    """

    field = metadata_filter['field']
    operator = metadata_filter['operator']
    value = metadata_filter['value']

    # Map the operator to SQL syntax
    operator_map = {
        'equals': '=',
        'not_equals': '!=',
        'in': 'IN',
        'not_in': 'NOT IN',
        'greater_than': '>',
        'less_than': '<',
        'greater_than_equals': '>=',
        'less_than_equals': '<=',
    }

    # Ensure the operator is valid
    if operator not in operator_map:
        raise ValueError(f"Unsupported operator: {operator}")

    sql_operator = operator_map[operator]

    # Handle different types of values
    if isinstance(value, list):
        # Convert list to a tuple for SQL IN expressions
        value_placeholder = f"({', '.join(['%s'] * len(value))})"
    else:
        # Single value placeholder
        value_placeholder = "%s"

    # Construct the SQL filter expression
    if operator in ['in', 'not_in']:
        filter_expression = f"metadata->>'{field}' {sql_operator} {value_placeholder}"
    else:
        filter_expression = f"metadata->>'{field}' {sql_operator} {value_placeholder}"

    return filter_expression


class PostgresVectorDB(VectorDB):
    def __init__(self, kb_id: str, username: str, password: str, database: str, host: str = "localhost", port: int = 5432, vector_dimension: int = 768):
        self.kb_id = kb_id
        self.table_name = f'{kb_id}_vectors'
        self.index_name = f'{kb_id}_embedding_index'
        self.username = username
        self.password = password
        self.database = database
        self.host = host
        self.port = port
        self.vector_dimension = vector_dimension

        # Create the extension if it doesn't exist
        conn = psycopg2.connect(
            dbname=database,
            user=username,
            password=password,
            host=host,
            port=port
        )
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.commit()

        # Import register_vector only when needed
        from pgvector.psycopg2 import register_vector
        register_vector(conn)

        from psycopg2 import sql

        cur.execute(
            sql.SQL(
                "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = {})")
            .format(sql.Literal(self.table_name))
        )
        exists = cur.fetchone()[0]
        print("exists", exists)

        # Create the table for this kb id if it doesn't exist
        if not exists:
            cur.execute(
                sql.SQL(
                    "CREATE TABLE {} (id TEXT PRIMARY KEY, metadata JSONB, embedding vector(%s))")
                .format(sql.Identifier(self.table_name)),
                [vector_dimension]
            )
            conn.commit()

            # Create the index
            cur.execute(
                sql.SQL(
                    """
                    CREATE INDEX {} ON {} USING hnsw(embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """)
                .format(
                    sql.Identifier(self.index_name),
                    sql.Identifier(self.table_name)
                )
            )
            conn.commit()

        conn.close()

    def get_num_vectors(self):
        conn = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port
        )

        try:
            from psycopg2 import sql

            cur = conn.cursor()
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").FORMAT(sql.Identifier(self.table_name)))
            count = cur.fetchone()[0]
        finally:
            conn.close()
        return count

    def add_vectors(self, vectors: Sequence[Vector], metadata: Sequence[ChunkMetadata]):

        conn = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port
        )
        cur = conn.cursor()

        vectors = np.array(vectors)
        # Create the ids from the doc_id and chunk_index
        ids = [
            f"{content['doc_id']}_{content['chunk_index']}" for content in metadata]
        data_to_insert = [(id, json.dumps(content), embedding)
                          for id, content, embedding in zip(ids, metadata, vectors)]

        from psycopg2 import sql
        insert_sql = sql.SQL("INSERT INTO {} (id, metadata, embedding) VALUES (%s, %s, %s)").format(
            sql.Identifier(self.table_name)).as_string(cur)

        cur.executemany(insert_sql, data_to_insert)
        conn.commit()
        conn.close()

    def remove_document(self, doc_id):

        conn = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port
        )
        cur = conn.cursor()

        # Delete all vectors with the given doc_id
        condition = {"doc_id": doc_id}

        from psycopg2 import sql
        cur.execute(
            sql.SQL(
                "DELETE FROM {} WHERE metadata @> %s").format(sql.Identifier(self.table_name)),
            [json.dumps(condition)]
        )

        conn.commit()
        conn.close()

    def search(self, query_vector: list, top_k: int = 10, metadata_filter: Optional[MetadataFilter] = None):

        conn = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port
        )
        cur = conn.cursor()

        query_vector = np.array(query_vector)

        if metadata_filter:
            filter_expression = format_metadata_filter(metadata_filter)

        from psycopg2 import sql
        if metadata_filter:
            filter_value = metadata_filter['value']

            query = sql.SQL("""
                SELECT metadata, embedding, 1 - (embedding <=> %s) AS cosine_similarity
                FROM {} 
                WHERE {} 
                ORDER BY cosine_similarity DESC 
                LIMIT %s
            """).format(
                sql.Identifier(self.table_name),
                sql.SQL(filter_expression)
            )

            if isinstance(filter_value, list):
                params = (query_vector, *filter_value, top_k)
            else:
                params = (query_vector, filter_value, top_k)

            cur.execute(query, params)
        else:
            query = sql.SQL("""
                SELECT metadata, embedding, 1 - (embedding <=> %s) AS cosine_similarity
                FROM {}
                ORDER BY cosine_similarity DESC
                LIMIT %s
            """).format(sql.Identifier(self.table_name))
            cur.execute(query, (query_vector, top_k))

        results = cur.fetchall()
        formatted_results: list[VectorSearchResult] = []
        for row in results:
            metadata, embedding, cosine_similarity = row

            formatted_results.append(
                VectorSearchResult(
                    doc_id=metadata["doc_id"],
                    vector=embedding,
                    metadata=metadata,
                    similarity=cosine_similarity,
                )
            )

        conn.close()

        return formatted_results

    def delete(self):
        # Delete the table
        conn = psycopg2.connect(
            dbname=self.database,
            user=self.username,
            password=self.password,
            host=self.host,
            port=self.port
        )

        from psycopg2 import sql
        cur = conn.cursor()
        cur.execute(sql.SQL("DROP TABLE {}").format(
            sql.Identifier(self.table_name)))
        conn.commit()
        conn.close()

    def to_dict(self):
        return {
            **super().to_dict(),
            "kb_id": self.kb_id,
            "username": self.username,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "vector_dimension": self.vector_dimension
        }
