################################################################################
######################### Import Requisite Libraries ###########################
import os
import sys
import typer
import pandas as pd

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

# sys.path.insert(0, "core")

################################################################################

from core.constants import (
    var_index,
    creat_var,
    time_var,
    time_var_corr,  
    creat_var_corr,
    preproc_run_name,
    exp_artifact_name,
)

# import all user-defined functions and constants
from core.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    safe_to_numeric,
)

app = typer.Typer()


@app.command()
def main(
    input_data_file: str = "./data/processed/df.parquet",
    output_data_file: str = "./data/processed/df_sans_zero.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage (e.g., 'training' or 'inference').
    """
    ############################################################################
    # Step 1. Read the input data file
    ############################################################################

    df = pd.read_parquet(input_data_file)

    # Set index if not already set
    if df.index.name != var_index:
        try:
            df.set_index(var_index, inplace=True)
            print(f"\nIndex set to '{var_index}'.")
        except KeyError:
            print(
                f"Warning: '{var_index}' not found in columns - "
                "proceeding with default integer index."
            )
    else:
        print(f"Index '{var_index}' already set - skipping.")

    ###########################################################################
    # Step 3. Rename Columns for Consistency
    ###########################################################################
    rename_map = {creat_var: creat_var_corr, time_var: time_var_corr}
    renamed = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=renamed, inplace=True)

    for old, new in renamed.items():
        print(f"Renamed '{old}' to '{new}'.")
    if stage == "training":

        df_object = df.select_dtypes("object")
        print()
        print(
            "The following columns have strings and may need to be removed from "
            "modeling and/or otherwise transformed with `categorical_transformer` "
            f"\nas handled accordingly in the `config.py` file. This list is stored "
            f"as an artifact in MLflow for future reference if necessary for "
            f"retrieval at a later time. \n \n"
            f"There are {df_object.shape[1]} string columns:\n \n"
            f"{df_object.columns.to_list()}. \n "
        )

        ########################################################################
        # Step 2. String Columns Handling
        ########################################################################
        # String columns are identified and should be removed before modeling
        # because machine learning models typically require numerical inputs.
        # Keeping string columns in the dataset may lead to errors or
        # unintended behavior unless explicitly encoded.
        #
        # To ensure consistency between training and inference,
        # we save the list of string columns and track it using MLflow.
        ########################################################################

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ########################################################################
        # Step 3. Save and Log String Column List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ########################################################################

        # Dump the string_cols_list into a pickle file for future reference
        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        # Log the string column list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )

    ############################################################################
    ###################### Re-engineering Selected Features ####################
    ############################################################################

    ########################################################################
    # Step 4. Ensure Numeric Data and Feature Engineering
    ########################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    # Perform necessary feature transformations (if and as applicable), such as:
    # - Deriving weight in pounds from kilograms
    # - Calculating height in feet using BMI and weight
    # - Dropping redundant features to prevent overfitting
    ########################################################################

    # Convert possible numeric columns to actual numeric types
    df = df.apply(lambda x: safe_to_numeric(x))


    ################################################################################
    # Step 5. Zero Variance Columns
    ################################################################################

    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.

    ################################################################################

    if stage == "training":
        # Extract numeric columns to compute variance and identify
        # zero-variance features
        numeric_cols = df.select_dtypes(include=["number"]).columns
        var_indf = df[numeric_cols].var()

        # identify zero variance columns
        zero_var = var_indf[var_indf == 0]
        # capture zero-variance cols in list
        zero_varlist_list = list(zero_var.index)

        print("*" * 80)
        print(f"Zero-variance columns: {zero_varlist_list}")
        print("*" * 80)

        ########################################################################
        # Step 6. Save and Log Zero Variance Columns List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        ########################################################################

        dumpObjects(
            zero_varlist_list,
            os.path.join(data_path, "zero_varlist_list.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
            obj=zero_varlist_list,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Zero Variance Columns List
        ########################################################################

        # load zero_var_list
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    ########################################################################
    # Step 7. Remove zero variance cols from main df, and assign to new var
    # df_sans_zero
    ########################################################################
    df_sans_zero = df.drop(columns=zero_varlist_list)

    print(f"Sans Zero Var Shape: {df_sans_zero.shape}")

    print()
    print(f"Original shape: {df.shape[1]} columns.")
    print(f"Reduced by {df.shape[1]-df_sans_zero.shape[1]} zero variance columns.")
    print(f"Now there are {df_sans_zero.shape[1]} columns.")
    print()

    ############################################################################
    # Step 8. Save Processed Data
    ############################################################################

    # Save out the dataframe to parquet file
    print(df_sans_zero.shape)
    df_sans_zero.reset_index().to_parquet(output_data_file)


if __name__ == "__main__":
    app()