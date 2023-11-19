import argparse
import pandas as pd
from utils import translate_sentences 

def main():
    parser = argparse.ArgumentParser(description="Translate specified columns of an Excel file.")
    
    parser.add_argument('--excel_file_path', type=str, required=True, 
                        help='Path to the Excel file.')
    parser.add_argument('--column_names_to_translate', type=str, nargs='+', required=True,
                        help='List of column names to translate.')

    args = parser.parse_args()

    data = translate_sentences(excel_file_path=args.excel_file_path, 
                               column_names=args.column_names_to_translate)

    output_file = f"./data/Processed_{args.excel_file_path.split('/')[-1]}"
    data.to_excel(output_file)
    print(f"Translated data saved to {output_file}")

if __name__ == '__main__':
    main()
