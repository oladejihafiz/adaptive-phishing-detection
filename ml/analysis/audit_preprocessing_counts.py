"""
READ-ONLY audit of ml/prepare_data.py: counts how many records each
preprocessing step removes, so the drop from the source total to the
processed corpus can be explained exactly.

Run from the project root:   python -m ml.audit_preprocessing_counts
It imports SOURCES, RAW and normalize_url from ml.prepare_data so that it
uses the same rules. It writes nothing except results/metrics/preprocessing_step_counts.csv.
"""
from pathlib import Path
import pandas as pd
from ml.prepare_data import SOURCES, RAW, normalize_url, ROOT

def main():
    rows = []
    parts = []
    within_dups_total = 0
    for filename, url_col, label_col, mapping in SOURCES:
        df = pd.read_csv(RAW / filename, usecols=[url_col, label_col], low_memory=False)
        raw = len(df)
        df.columns = ['url', 'original_label']
        df['label'] = df['original_label'].map(mapping)
        label_unmapped = int(df['label'].isna().sum())
        df['url'] = df['url'].map(normalize_url)
        url_invalid = int(df['url'].isna().sum())
        df = df.dropna(subset=['url', 'label'])[['url', 'label']]
        df['label'] = df['label'].astype(int)
        usable = len(df)
        within = usable - len(df.drop_duplicates(['url', 'label']))
        within_dups_total += within
        rows.append({'step': f'{filename}: raw rows', 'count': raw})
        rows.append({'step': f'{filename}: rows with unmapped/missing label', 'count': label_unmapped})
        rows.append({'step': f'{filename}: rows with missing/invalid URL', 'count': url_invalid})
        rows.append({'step': f'{filename}: usable rows (what prepare_data.py prints)', 'count': usable})
        rows.append({'step': f'{filename}: exact (url,label) duplicates within this file', 'count': within})
        parts.append(df)

    all_data = pd.concat(parts, ignore_index=True)
    n0 = len(all_data)
    rows.append({'step': 'Sum of usable rows over the three files', 'count': n0})

    d1 = all_data.drop_duplicates(['url', 'label'])
    rows.append({'step': 'Removed by drop_duplicates(url,label): total', 'count': n0 - len(d1)})
    rows.append({'step': '  of which duplicates inside a single file', 'count': within_dups_total})
    rows.append({'step': '  of which duplicates across different files', 'count': (n0 - len(d1)) - within_dups_total})

    conflict = d1.groupby('url')['label'].nunique()
    conflict_urls = set(conflict[conflict > 1].index)
    d2 = d1[~d1['url'].isin(conflict_urls)]
    rows.append({'step': 'Conflicting-label URLs (distinct URLs)', 'count': len(conflict_urls)})
    rows.append({'step': 'Rows removed because their URL has conflicting labels', 'count': len(d1) - len(d2)})

    d3 = d2.drop_duplicates('url')
    rows.append({'step': 'Removed by final drop_duplicates(url)', 'count': len(d2) - len(d3)})

    frsc_path = RAW / 'dataset_5.csv'
    frsc_removed = 0
    if frsc_path.exists():
        frsc = pd.read_csv(frsc_path, usecols=['url'])
        frsc_urls = set(frsc['url'].map(normalize_url).dropna())
        d4 = d3[~d3['url'].isin(frsc_urls)]
        frsc_removed = len(d3) - len(d4)
    else:
        d4 = d3
    rows.append({'step': 'Rows removed because the URL is in dataset_5 (FRSC)', 'count': frsc_removed})
    rows.append({'step': 'Final processed corpus (before the 70:15:15 split)', 'count': len(d4)})
    rows.append({'step': 'Total reduction from the sum of usable rows', 'count': n0 - len(d4)})

    out = pd.DataFrame(rows)
    pd.set_option('display.max_colwidth', 100)
    print(out.to_string(index=False))
    res = ROOT / 'results' / 'metrics'
    res.mkdir(parents=True, exist_ok=True)
    out.to_csv(res / 'preprocessing_step_counts.csv', index=False)
    print('\nSaved:', res / 'preprocessing_step_counts.csv')

if __name__ == '__main__':
    main()