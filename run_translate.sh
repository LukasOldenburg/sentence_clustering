#!/bin/bash

# Script to run the Python script for translating specified columns of an Excel file.

# Define default values for the arguments
EXCEL_FILE_PATH="./data/example_data_translation.xlsx" # Replace with your default Excel file path
COLUMN_NAMES_TO_TRANSLATE=("Frage" "Antwort") # Replace with your default column names to translate

# Run the Python script with the arguments
python ./translate_sentences.py \
    --excel_file_path "$EXCEL_FILE_PATH" \
    --column_names_to_translate "${COLUMN_NAMES_TO_TRANSLATE[@]}"

