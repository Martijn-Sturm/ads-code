import polars as pl

# import numpy as np


# def pearsonr_pval(df: pl.DataFrame):
#     names = pl.DataFrame(df.columns, schema=["name"])
#     return names.hstack(
#         pl.DataFrame(
#             ([np.correlate for x in df.columns] for y in df.columns),
#             schema=df.columns,
#         )
#     )


def spearman_correlation(df: pl.DataFrame, col: str):
    return df.select(
        [
            pl.corr(pl.col(col), pl.col(feat), method="spearman").alias(feat)
            for feat in df.columns
        ]
    )
