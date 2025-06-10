import json
from pathlib import Path
from tqdm import tqdm
from typing import List
import pandas as pd

#file = Path("data") / "Bugs" / "linux_bugs.csv"
file = "data/BGC/BlurbGenreCollection_EN_dev.txt"

_output_data = Path("data") / "Amazon" / "samples.jsonl"
#Path("data") / "BGC" / "BlurbGenreCollection_EN_train.txt"
def parse_jsonl(file_path: Path) -> List[dict]:
    with open(file_path, mode="r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data
def parse_txt(file_path: Path) -> List[dict]:
    with open(file_path, mode="r", encoding='utf-8') as f:
        data = f.readlines()
    return data
def parse_csv(file_path: Path, lines: int) -> List[dict]:
    data = pd.read_csv(file_path, sep="\t")
    for i in range(lines):
        print(f"Line {i}: {data.iloc[i].to_dict()}")

def merge_jsonl_files(input_files: List[Path]) -> None:
    _output_data.parent.mkdir(parents=True, exist_ok=True)
    with open(_output_data, "w", encoding="utf-8") as out_f:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as in_f:
                for line in tqdm(in_f):
                    out_f.write(line)

if __name__ == "__main__":
    data = parse_txt(file)
    print(f"Parsed {len(data)} lines from {file}")
    print(f"Sample book data: \n{data[:100] if data else 'No data available'}")
    print("Done processing book.")
    #data = parse_csv(file, 2)
    #merge_jsonl_files([Path("data") / "Amazon" / "amazon_train.jsonl", Path("data") / "Amazon" / "amazon_test.jsonl"])
