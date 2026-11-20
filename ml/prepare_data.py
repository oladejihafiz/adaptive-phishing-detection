from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset' / 'raw'
OUT = ROOT / 'dataset' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    ('dataset_1.csv', 'URL', 'Label', {1: 1, 0: 0}),
    ('dataset_2.csv', 'url', 'type', {'phishing': 1, 'legitimate': 0}),
    ('dataset_4.csv', 'URL', 'Label', {'bad': 1, 'good': 0}),
]

def normalize_url(value):
    if pd.isna(value):
        return None
    url = str(value).strip()
    if not url or any(c.isspace() for c in url):
        return None
    if '://' not in url:
        url = 'http://' + url
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in ('http', 'https') or not parsed.hostname:
            return None
        return url
    except ValueError:
        return None

def features(url):
    p = urlsplit(url)
    host = (p.hostname or '').lower()
    path = p.path or ''
    try:
        is_ip = int(ipaddress.ip_address(host) is not None)
    except ValueError:
        is_ip = 0
    return {
        'url_length': len(url),
        'hostname_length': len(host),
        'path_length': len(path),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_at': url.count('@'),
        'num_question': url.count('?'),
        'num_equals': url.count('='),
        'num_slashes': url.count('/'),
        'is_https': int(p.scheme.lower() == 'https'),
        'has_ip_host': is_ip,
        'num_host_parts': len(host.split('.')),
        'has_punycode': int('xn--' in host),
        'suspicious_words': sum(w in url.lower() for w in ('verify', 'login', 'account', 'secure', 'update', 'password', 'bank')),
    }

def main():
    parts = []
    for filename, url_col, label_col, mapping in SOURCES:
        path = RAW / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing file: {path}')
        df = pd.read_csv(path, usecols=[url_col, label_col], low_memory=False)
        df.columns = ['url', 'original_label']
        df['label'] = df['original_label'].map(mapping)
        df['url'] = df['url'].map(normalize_url)
        df = df.dropna(subset=['url', 'label'])[['url', 'label']]
        df['label'] = df['label'].astype(int)
        print(f'{filename}: {len(df):,} usable rows')
        parts.append(df)

    all_data = pd.concat(parts, ignore_index=True)
    all_data = all_data.drop_duplicates(['url', 'label'])
    conflict = all_data.groupby('url')['label'].nunique()
    conflict_urls = set(conflict[conflict > 1].index)
    print(f'Conflicting-label URLs excluded: {len(conflict_urls):,}')
    all_data = all_data[~all_data['url'].isin(conflict_urls)]
    all_data = all_data.drop_duplicates('url').reset_index(drop=True)

    # Keep FRSC samples outside all initial training/validation/test files.
    frsc_path = RAW / 'dataset_5.csv'
    if frsc_path.exists():
        frsc = pd.read_csv(frsc_path, usecols=['url'])
        frsc_urls = set(frsc['url'].map(normalize_url).dropna())
        all_data = all_data[~all_data['url'].isin(frsc_urls)].copy()

    train, remainder = train_test_split(all_data, test_size=0.30,
                                        stratify=all_data['label'], random_state=42)
    validation, test = train_test_split(remainder, test_size=0.50,
                                        stratify=remainder['label'], random_state=42)
    for name, subset in [('train', train), ('validation', validation), ('test', test)]:
        numeric = pd.DataFrame.from_records(
            subset['url'].map(features).tolist()
           )
        numeric['label'] = subset['label'].to_numpy()
        numeric.to_csv(OUT / f'{name}.csv', index=False)
        subset[['url', 'label']].to_csv(OUT / f'{name}_urls.csv', index=False)
        print(f'{name}: {len(subset):,} rows; phishing={subset.label.sum():,}')
    print('Preprocessing complete. Files saved in dataset/processed/')

if __name__ == '__main__':
    main()

