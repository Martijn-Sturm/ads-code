import polars

csv_properties = {"has_header": True, "quote_char": r'"'}


def read_csv(table_name: str):
    path = f"data/{table_name}.csv"
    return polars.read_csv(path, **csv_properties)
