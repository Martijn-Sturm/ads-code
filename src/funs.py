import polars as pl
import math
import typing


def t_spearman(spearman_r: float, sample_size: int):
    return spearman_r * math.sqrt(sample_size - 2) / math.sqrt(1 - (spearman_r**2))


def t_spearman_listwise(r: typing.Sequence, n: typing.Sequence):
    return [t_spearman(r, n) for r, n in zip(r, n)]


def spearman_correlation_df(df: pl.DataFrame, col: str):
    spearman = df.select(
        [
            pl.corr(pl.col(col), pl.col(feat), method="spearman").alias(feat)
            for feat in df.columns
            if feat != col
        ]
    ).melt(variable_name="column", value_name="r")
    sample_size = (
        df.null_count()
        .melt(variable_name="column", value_name="null_count")
        .with_columns((pl.lit(len(df)) - pl.col("null_count")).alias("n"))
    )

    df = spearman.join(sample_size, on="column")
    t = [t_spearman(r, n) for r, n in zip(df.get_column("r"), df.get_column("n"))]
    return df.with_columns(pl.Series("t", t)).with_columns(
        (pl.col("t") / pl.lit(len(df))).alias("t_corrected")
    )


def spearman(df: pl.DataFrame, col: str):
    data = [
        {
            "column": feat,
            "r": pl.corr(col, feat, method="spearman"),
            "n": len(df) - df.get_column(feat).null_count(),
        }
        for feat in df.columns
    ]
    return data
