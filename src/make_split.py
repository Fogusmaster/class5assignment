import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def class_distribution_report(df: pd.DataFrame, target_col: str, name: str) -> str:
    total = len(df)
    counts = df[target_col].value_counts(dropna=False).sort_index()
    lines = [f"{name} rows: {total}"]
    for cls, count in counts.items():
        pct = (count / total * 100) if total else 0.0
        lines.append(f"  {target_col}={cls}: {count} ({pct:.2f}%)")
    return "\n".join(lines)


def resolve_column_name(df: pd.DataFrame, requested_name: str) -> str | None:
    if requested_name in df.columns:
        return requested_name
    lowered = {col.lower(): col for col in df.columns}
    return lowered.get(requested_name.lower())


def split_data(
    input_path: Path,
    output_dir: Path,
    target_col: str,
    group_col: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    df = pd.read_csv(input_path)

    resolved_target_col = resolve_column_name(df, target_col)
    if not resolved_target_col:
        raise ValueError(
            f"Target column '{target_col}' not found in {input_path}")

    resolved_group_col = resolve_column_name(
        df, group_col) if group_col else None

    use_group_split = bool(resolved_group_col)

    if use_group_split:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(
            splitter.split(df, y=df[resolved_target_col],
                           groups=df[resolved_group_col])
        )
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[resolved_target_col],
            random_state=random_state,
        )
        train_df = train_df.copy()
        test_df = test_df.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, test_df, use_group_split, resolved_target_col, resolved_group_col


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test CSV files.")
    parser.add_argument(
        "--input", default="data/ass5data.csv", help="Input CSV path")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory")
    parser.add_argument("--target", default="prodtaken",
                        help="Target column name")
    parser.add_argument("--group-col", default="",
                        help="Optional group column name")
    parser.add_argument("--test-size", type=float,
                        default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_df, test_df, used_group_split, resolved_target_col, resolved_group_col = split_data(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        target_col=args.target,
        group_col=args.group_col.strip() or None,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print("Split complete.")
    print(class_distribution_report(train_df, resolved_target_col, "Train"))
    print(class_distribution_report(test_df, resolved_target_col, "Test"))

    if used_group_split:
        train_groups = set(train_df[resolved_group_col].dropna().unique())
        test_groups = set(test_df[resolved_group_col].dropna().unique())
        overlap = train_groups.intersection(test_groups)
        print(f"Group split used: True")
        print(f"Overlapping group IDs: {len(overlap)}")
    else:
        print("Group split used: False (stratified random split)")


if __name__ == "__main__":
    main()
