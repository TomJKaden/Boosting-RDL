"""
Converts task statistics to Latex format.
"""

import pandas as pd


def main():
    df = pd.read_csv("binary_stats.csv", index_col=0)
    convert(df, 0)
    df = pd.read_csv("multiclass_stats.csv", index_col=0)
    convert(df, 1)
    df = pd.read_csv("regression_stats.csv", index_col=0)
    convert(df, 2)


def convert(df: pd.DataFrame, t: int):
    for row in df.itertuples(index=False):
        print(
            f"{row[0].replace('rel-', '')} & {shorten(row[1])} & {row[2][0].upper()} & {row[3]} & {round(row[4], 2)} & {round(row[5], 2)} & {row[6]} & {round(row[7], 2)}",
            end="",
        )
        if t == 0:
            print(f" & {round(row[10], 3)}\\\\")
        if t == 1:
            print(f" & {row[10]} & {round(row[11], 1)} & {round(row[12], 3)}\\\\")
        if t == 2:
            print(
                f" & {round(row[10], 1)} & {round(row[11], 2)} & {round(row[12], 2)} & {round(row[13], 2)} & {round(row[14], 2)} & {round(row[15], 2)}\\\\"
            )
    print("\\uzlhline")


def shorten(s: str):
    split = s.split("-")
    return split[0][:4] + "-" + split[1][:4]


if __name__ == "__main__":
    main()
