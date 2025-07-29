
import numpy as np
import pandas as pd

bert_pred_y_fp = "dumps/BERT/bert_multilabel_AMZ_concat_cls_3/run_2025-07-09_09-35-30/all_folds_pred_2025-07-11_11-09-38.csv"
bert_match_pred_y_fp = "dumps/BERT_MATCH/bert_multilabel_AMZ_concat_cls_3/run_2025-07-11_14-08-25/all_folds_pred_2025-07-15_13-42-12.csv"
hbgl_pred_y_fp = ""

#check mutual exclusivity errors (2 leaves or more)

def count_me(predictions, n_leaves):
    # Select last leaf columns (species-level predictions)
    
    leaf_preds = predictions[:, -n_leaves:]
    # Count how many 1s (i.e., positive predictions) per row
    leaf_counts = np.sum(leaf_preds, axis=1)
    # Identify rows with incorrect number of predictions
    non_exclusive_rows = np.where(leaf_counts > 1)[0]
    zero_rows = np.where(leaf_counts == 0)[0]
    n = 5
    if len(non_exclusive_rows) > n and len(zero_rows) > n:
        for i in range(n):
            print(f"no.{i}:")
            print(f"  non_exclusive_row {non_exclusive_rows[i]} → {predictions[non_exclusive_rows[i]]}")
            print(f"  zero_row          {zero_rows[i]} → {predictions[zero_rows[i]]}")    # Count and optionally display some examples
    
    return non_exclusive_rows, zero_rows
    
def compare_hierarchy_violations(predictions, ohe_dict): #convert to tuples for hashability -> faster?
    # Convert all values to tuples once and store in a set
    allowed_set = {tuple(v.values()) for v in ohe_dict.values()}
    #print(allowed_set)

    count = 0
    for i in predictions:
        i_tuple = tuple(i)  # Convert prediction to tuple
        if i_tuple not in allowed_set:
            count += 1
    return count

ohe_dict_from_csv = "amazon_tax_one_hot.csv" # Replace with csv path
if "amazon" in ohe_dict_from_csv:
    n_leaves = 25
elif "bgc" in ohe_dict_from_csv:
    n_leaves = 120
elif "wos" in ohe_dict_from_csv:
    n_leaves = 138
else:
    raise ValueError("Unknown OHE dictionary csv")

#include ME violations as hierarchy violations


def main(): 
    df = pd.read_csv(bert_match_pred_y_fp)
    # Convert to dictionary
    ohe_dict = pd.read_csv(ohe_dict_from_csv, index_col=0).astype(int).to_dict(orient='index')

    # Convert to NumPy array
    predictions = df.values
    non_exclusive_rows, zero_rows = count_me(predictions, n_leaves)

    hv_count = compare_hierarchy_violations(predictions, ohe_dict)
        
    print(f"Total HV: {hv_count}")
    print(f"ME violations: {len(non_exclusive_rows)}\nZero predictions: {len(zero_rows)}")
    print(f"Other HV (Total HV minus ME and zero): {hv_count - len(non_exclusive_rows) - len(zero_rows)}")
if __name__ == "__main__":
    main()