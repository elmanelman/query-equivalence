import collections
import dataclasses
import secrets
import typing

import pglast
import pglast.printers
import psycopg2

from equivalence import values


@dataclasses.dataclass(frozen=True)
class ConnectionParameters:
    """Represents regular PostgreSQL connection parameters"""

    user: str
    password: str
    dbname: str
    host: str = "localhost"
    port: int = 5432


@dataclasses.dataclass(frozen=True)
class ColumnInfo:
    """Represents information about a specific column"""

    column_name: str
    column_type: str
    is_nullable: bool
    ordinal_position: int


class PlanGenerator:
    def __deconfigure_planner(self):
        parameters_to_reset = [
            "enable_indexscan",
            "enable_bitmapscan",
            "enable_tidscan",
            "enable_hashjoin",
            "enable_mergejoin",
            "enable_material",
        ]

        with self.connection.cursor() as cursor:
            for parameter in parameters_to_reset:
                cursor.execute(f"set {parameter} to default")

            cursor.execute("set max_parallel_workers_per_gather to default")

    def __configure_planner(self):
        parameters_to_off = [
            "enable_indexscan",
            "enable_bitmapscan",
            "enable_tidscan",
            "enable_hashjoin",
            "enable_mergejoin",
            "enable_material",
        ]

        with self.connection.cursor() as cursor:
            for parameter in parameters_to_off:
                cursor.execute(f"set {parameter} = 'off'")

            cursor.execute("set max_parallel_workers_per_gather = 0")

    def __init__(
        self,
        connection_parameters: typing.Union[ConnectionParameters, str],
        schema_script=None,
    ):
        if isinstance(connection_parameters, str):
            self.connection = psycopg2.connect(connection_parameters)
        else:
            self.connection = psycopg2.connect(
                **dataclasses.asdict(connection_parameters)
            )

        self.schema_name = "public"
        self.schema_script = schema_script
        if schema_script is not None:
            self.use_schema(schema_script)
        self.schema_info = None

        self.values_relations = {}

        self.__configure_planner()

    def use_schema(self, schema_script):
        """Creates a new schema from the script

        :param schema_script: DDL script to use
        """
        if self.schema_script is not None:
            current_schema_fingerprint = pglast.fingerprint(self.schema_script)
            given_schema_fingerprint = pglast.fingerprint(schema_script)
            if current_schema_fingerprint == given_schema_fingerprint:
                return

        with self.connection.cursor() as cursor:
            self.schema_name = f"qe_{secrets.token_hex(4)}"
            self.schema_script = schema_script

            cursor.execute(f"create schema {self.schema_name}")
            cursor.execute(f"set search_path = '{self.schema_name}'")
            cursor.execute(schema_script)

    def get_schema_info(self, schema_name=None):
        """Gets information about the schema

        :param schema_name: schema name
        :return: dictionary with schema information
        """
        if schema_name is None:
            if self.schema_info is not None:
                return self.schema_info

            if self.schema_name is not None:
                schema_name = self.schema_name
            else:
                raise ValueError("schema name not specified")

        query = """
            with constraints_info as (
                select tc.table_name, ccu.column_name, tc.constraint_type
                from information_schema.table_constraints tc
                join information_schema.constraint_column_usage ccu
                    using (constraint_schema, constraint_name)
            )
            select
                t.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.ordinal_position,
                ci.constraint_type
            from information_schema.tables t
            join information_schema.columns c
                on t.table_name = c.table_name
            left join constraints_info ci
                on t.table_name = ci.table_name and c.column_name = ci.column_name
            where t.table_schema = 'public'
            order by t.table_name, c.ordinal_position
        """

        schema_info = collections.defaultdict(dict)

        self.__deconfigure_planner()

        with self.connection.cursor() as cursor:
            cursor.execute(query, {"schema_name": schema_name})

            schema_info_rows = cursor.fetchall()
            for row in schema_info_rows:
                (
                    table_name,
                    column_name,
                    data_type,
                    is_nullable,
                    ordinal_position,
                    constraint_type,
                ) = row
                bool_map = {"NO": False, "YES": True}
                column_info = ColumnInfo(
                    column_name,
                    data_type,
                    bool_map[is_nullable],
                    ordinal_position,
                )
                schema_info[table_name][column_name] = column_info
                if constraint_type == "PRIMARY KEY":
                    schema_info[table_name]["PRIMARY KEY"] = column_info

            cursor.execute("set enable_indexscan = 'off'")

        self.schema_info = dict(schema_info)

        self.__configure_planner()

        return self.schema_info

    def drop_schema(self):
        """Drops current schema"""
        with self.connection.cursor() as cursor:
            cursor.execute("set search_path = 'public'")
            cursor.execute(f"drop schema {self.schema_name} cascade")

            self.schema_name = "public"

    def get_json(self, query):
        """Requests a query execution plan from the PostgreSQL planner.

        :param query: selection query
        :return: dictionary of the execution plan tree
        """
        with self.connection.cursor() as cursor:
            prepared_query, definitions = values.prepare_query(query)

            if len(definitions) > 0:
                query = prepared_query

                if self.schema_info is None:
                    self.schema_info = {}

                for relation, definition in definitions.items():
                    create_query = f"""
                        create table {relation} as
                        select * from ({definition}) as {relation}"""
                    cursor.execute(create_query)

                    cursor.execute(f"select * from {relation}")
                    self.values_relations[relation] = cursor.fetchall()

                    if relation not in self.schema_info:
                        self.schema_info[relation] = {}

                    first_row = self.values_relations[relation][0]
                    for i, value in enumerate(first_row):
                        column_name = f"column{i + 1}"
                        self.schema_info[relation][column_name] = ColumnInfo(
                            column_name, type(value).__name__, False, i + 1
                        )

            explain_query = f"explain (format json, verbose) {query}"

            cursor.execute(explain_query)
            plan_row = cursor.fetchone()

            for relation in definitions:
                cursor.execute(f"drop table {relation}")

            return plan_row[0][0]["Plan"]
