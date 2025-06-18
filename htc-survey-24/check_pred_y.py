
import numpy as np
import pandas as pd

pred_y_file = "pred_y/20250514-150737_250508_model-d/predicted_test_ConstrainedFFNNModel_oeFalse_250508_model-d_20250514-150737.csv"




#Check for mutual exclusivity violations
def count_me(predictions):
    # Select last 200 columns (species-level predictions)
    species_preds = predictions[:, -200:]

    # Count how many 1s (i.e., positive predictions) per row
    species_counts = np.sum(species_preds, axis=1)

    # Identify rows with incorrect number of predictions
    non_exclusive_rows = np.where(species_counts > 1)[0]
    zero_rows = np.where(species_counts == 0)[0]

    # Count and optionally display some examples
    return non_exclusive_rows, zero_rows
    
def compare_hierarchy_violations(predictions, ohe_dict): #convert to tuples for hashability -> faster?
    # Convert all values to tuples once and store in a set
    allowed_set = {tuple(v) for v in ohe_dict.values()}

    count = 0
    for i in predictions:
        i_tuple = tuple(i)  # Convert prediction to tuple
        if i_tuple not in allowed_set:
            count += 1
    return count

ohe_dict_from_csv = "amazon_tax_one_hot.csv" # Replace with csv path

#include ME violations as hierarchy violations


def main(): 
    df = pd.read_csv(pred_y_file)

    # Convert to NumPy array
    predictions = df.values
    non_exclusive_rows, zero_rows = count_me(predictions)

    hv_count = compare_hierarchy_violations(predictions, ohe_dict_from_csv)
    
    print(f"Total rows with ME violations: {len(non_exclusive_rows)}\nTotal rows with 0 species predicted: {len(zero_rows)}")
    print(f"Total rows with hierarchy violations: {hv_count}")
if __name__ == "__main__":
    main()