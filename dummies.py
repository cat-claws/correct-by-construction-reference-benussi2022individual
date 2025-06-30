import pandas as pd

def one_hot_integer_columns(df):
    df = df.copy()
    new_cols = []

    for col in df.columns:
        series = df[col]
        
        # Only act on strictly integer dtype
        if pd.api.types.is_integer_dtype(series):
            unique_vals = set(series.dropna().unique())

            # If only contains 0 and 1 → treat as binary, keep as-is
            if unique_vals <= {0, 1}:
                new_cols.append(col)
                continue

            # Else, one-hot encode
            one_hot = pd.get_dummies(series, prefix=col)
            insert_at = df.columns.get_loc(col)

            # Drop original and insert one-hot columns in place
            df = df.drop(columns=[col])
            for i, new_col in enumerate(one_hot.columns):
                df.insert(insert_at + i, new_col, one_hot[new_col])
            
            new_cols.extend(one_hot.columns)
        else:
            new_cols.append(col)

    # Reorder columns to reflect changes
    df = df[[col for col in new_cols if col in df.columns]]

    return df
