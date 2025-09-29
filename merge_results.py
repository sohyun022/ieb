#!/usr/bin/env python3
"""
merge_results.py

배치 결과(fan_origin_batch-*.tsv)들을 읽어 병합하고 간단한 요약을 출력합니다.
- TSV 파일들(pattern: {exp_id}/{pattern})을 모두 읽어 pandas DataFrame으로 합침
- 'response' 열을 숫자로 변환(변환 불가 값 -> NaN)
- 병합된 데이터프레임을 --out 으로 지정한 파일에 TSV로 저장
- 파일이 없거나 읽을 수 없으면 친절한 메시지 출력

사용법:
    python merge_results.py --exp_id test_run
    python merge_results.py --exp_id test_run --pattern "fan_origin_batch-*.tsv" --out merged.tsv
"""

import argparse
import glob
import os
import sys
from typing import List

import pandas as pd

def find_files(exp_id: str, pattern: str) -> List[str]:
    path_pattern = os.path.join(exp_id, pattern)
    files = sorted(glob.glob(path_pattern))
    return files

def read_file(path: str) -> pd.DataFrame:
    # 안전하게 읽기: 실패하면 예외를 발생시켜 상위에서 처리
    return pd.read_csv(path, sep='\t')

def main():
    parser = argparse.ArgumentParser(description="Merge TSV batch result files into one DataFrame.")
    parser.add_argument("--exp_id", "-e", required=True, help="experiment directory (e.g. test_run)")
    parser.add_argument("--pattern", "-p", default="fan_origin_batch-*.tsv", help="glob pattern for batch files inside <exp_id> dir")
    parser.add_argument("--out", "-o", default=None, help="output TSV file path to write merged results (default: <exp_id>/merged_results.tsv)")
    parser.add_argument("--head", type=int, default=5, help="print top N rows of merged dataframe (default 5)")
    parser.add_argument("--save", action="store_true", help="save merged file even if no files found? (ignored when no files)")
    args = parser.parse_args()

    exp_id = args.exp_id
    pattern = args.pattern
    out_path = args.out or os.path.join(exp_id, "merged_results.tsv")

    # 1) 파일 찾기
    files = find_files(exp_id, pattern)
    if not files:
        print(f"No files found for pattern: {os.path.join(exp_id, pattern)}", file=sys.stderr)
        print("Make sure the directory and pattern are correct. Exiting.")
        sys.exit(2)

    print(f"Found {len(files)} files. Reading...")

    df_list = []
    read_errors = 0
    for f in files:
        try:
            df = read_file(f)
            df['__source_file'] = os.path.basename(f)  # 추적용 컬럼 추가
            df_list.append(df)
        except Exception as e:
            read_errors += 1
            print(f"Warning: failed to read {f}: {e}", file=sys.stderr)

    if not df_list:
        print("No files were successfully read. Exiting.", file=sys.stderr)
        sys.exit(3)

    # 2) concat
    try:
        results_df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"Error concatenating dataframes: {e}", file=sys.stderr)
        sys.exit(4)

    # 3) response 열을 numeric으로 변환 (coerce -> invalid -> NaN)
    if 'response' in results_df.columns:
        results_df['response'] = pd.to_numeric(results_df['response'], errors='coerce')
    else:
        print("Warning: 'response' column not found in merged dataframe.", file=sys.stderr)

    # 4) 간단한 요약 출력
    n_rows, n_cols = results_df.shape
    print("=== Summary ===")
    print(f"Total files matched: {len(files)} (read errors: {read_errors})")
    print(f"Merged shape: {n_rows} rows x {n_cols} columns")
    print("Columns:", ", ".join(results_df.columns.tolist()))
    if 'response' in results_df.columns:
        n_response_na = results_df['response'].isna().sum()
        print(f"'response' NaN count: {n_response_na} ({(n_response_na/n_rows*100) if n_rows else 0:.2f}%)")
    # show head
    print("\n=== Top rows ===")
    print(results_df.head(args.head).to_string(index=False))

    # 5) 저장
    try:
        results_df.to_csv(out_path, sep='\t', index=False)
        print(f"\nMerged results saved to: {out_path}")
    except Exception as e:
        print(f"Failed to save merged results to {out_path}: {e}", file=sys.stderr)
        sys.exit(5)

if __name__ == "__main__":
    main()

