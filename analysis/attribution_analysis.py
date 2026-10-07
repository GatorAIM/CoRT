"""Generic attribution aggregation without patient-level output."""

from __future__ import annotations

import pandas as pd


def attribution_category_composition(
    attributions: pd.DataFrame,
    *,
    group_columns=("state", "model"),
    category_column="category",
    value_column="attribution",
) -> pd.DataFrame:
    """Return normalized absolute attribution mass by group and category."""
    frame = attributions.loc[:, [*group_columns, category_column, value_column]].copy()
    frame["absolute_attribution"] = pd.to_numeric(frame[value_column], errors="coerce").abs()
    summary = (
        frame.groupby([*group_columns, category_column], as_index=False)["absolute_attribution"]
        .sum()
    )
    totals = summary.groupby(list(group_columns))["absolute_attribution"].transform("sum")
    summary["proportion"] = summary["absolute_attribution"] / totals.where(totals.ne(0), 1)
    return summary


def positive_attribution_balance(
    attributions: pd.DataFrame,
    *,
    group_columns=("state", "model"),
    value_column="attribution",
) -> pd.DataFrame:
    """Calculate positive attribution mass divided by total absolute mass."""
    frame = attributions.loc[:, [*group_columns, value_column]].copy()
    frame["positive"] = frame[value_column].clip(lower=0)
    frame["absolute"] = frame[value_column].abs()
    summary = frame.groupby(list(group_columns), as_index=False).agg(
        positive_mass=("positive", "sum"),
        absolute_mass=("absolute", "sum"),
    )
    summary["positive_balance"] = summary["positive_mass"] / summary["absolute_mass"].where(
        summary["absolute_mass"].ne(0), 1
    )
    return summary

