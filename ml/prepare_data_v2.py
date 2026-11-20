"""Strict, auditable URL ingestion for the phishing-detection project.

This script does not extract features, split data, train models, or overwrite
existing processed data. It writes to dataset/processed_v2_audited/.
"""
from __future__ import annotations

from html import unescape
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import ipaddress
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # Project root when copied into ml/.
RAW_DIR = ROOT / "dataset" / "raw"
NEW_DIR = ROOT / "dataset" / "new_sources"
OUTPUT_DIR = ROOT / "dataset" / "processed_v2_audited"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Project labels: 1=phishing, 0=legitimate.
SELECTED_SOURCES = [
    {"file": RAW_DIR / "dataset_1.csv", "source": "dataset_1", "url_col": "URL", "label_col": "Label", "mapping": {"1": 1, "0": 0}},
    {"file": RAW_DIR / "dataset_2.csv", "source": "dataset_2", "url_col": "url", "label_col": "type", "mapping": {"phishing": 1, "legitimate": 0}},
    {"file": RAW_DIR / "dataset_4.csv", "source": "dataset_4", "url_col": "URL", "label_col": "Label", "mapping": {"bad": 1, "good": 0}},
    # PhiUSIIL uses 1=legitimate and 0=phishing; invert to project convention.
    {"file": NEW_DIR / "dataset_6.csv", "source": "phiusiil_dataset_6", "url_col": "URL", "label_col": "label", "mapping": {"1": 0, "0": 1}},
]
PHISHTANK_FILE = NEW_DIR / "dataset_10.csv"
FRSC_FILE = RAW_DIR / "dataset_5.csv"


def normalize_url_with_reason(value: object) -> tuple[str | None, str | None]:
    """Return (normalized URL, rejection reason); retain suspicious path/query text."""
    if pd.isna(value):
        return None, "missing_url"
    raw = unescape(str(value)).strip()
    if not raw:
        return None, "empty_url"
    # Reject controls/whitespace before urlsplit can silently discard them.
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        return None, "control_character"
    if re.search(r"\s", raw):
        return None, "embedded_whitespace"
    candidate = raw if "://" in raw else "http://" + raw
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme.lower() not in {"http", "https"}:
            return None, "unsupported_scheme"
        if not parsed.netloc or not parsed.hostname:
            return None, "missing_authority_or_hostname"
        authority = parsed.netloc.rsplit("@", 1)[-1]
        if "\\" in authority:
            return None, "backslash_in_authority"
        if any(ch in authority for ch in ("'", '"', "`")):
            return None, "quote_in_authority"
        host = parsed.hostname.lower()
        if "%" in host:
            return None, "percent_escape_in_hostname"
        port = parsed.port  # Raises ValueError for malformed ports.
        if port is not None and not 1 <= port <= 65535:
            return None, "invalid_port"
        try:
            ip_obj = ipaddress.ip_address(host)
            normalized_host = ip_obj.compressed.lower()
            if ip_obj.version == 6:
                normalized_host = f"[{normalized_host}]"
        except ValueError:
            # IDNA-convert internationalized domains, then validate DNS labels.
            dns_host = host[:-1] if host.endswith(".") else host
            if not dns_host:
                return None, "empty_hostname"
            try:
                ascii_host = dns_host.encode("idna").decode("ascii").lower()
            except UnicodeError:
                return None, "invalid_idna_hostname"
            if len(ascii_host) > 253:
                return None, "hostname_too_long"
            labels = ascii_host.split(".")
            if len(labels) < 2:
                return None, "single_label_hostname"
            for label in labels:
                if not label:
                    return None, "empty_hostname_label"
                if len(label) > 63:
                    return None, "hostname_label_too_long"
                if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label):
                    return None, "invalid_hostname_label"
            normalized_host = ascii_host
        # Preserve user-info if supplied, but never treat it as hostname text.
        userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
        netloc = userinfo + normalized_host + (f":{port}" if port is not None else "")
        return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment)), None
    except ValueError:
        return None, "url_parse_or_port_error"


def normalize_label(value: object, mapping: dict[str, int]) -> int | None:
    """Normalize source label text before applying its documented mapping."""
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    if key.endswith(".0") and key[:-2] in {"0", "1"}:
        key = key[:-2]
    return mapping.get(key)


def hostname_of(url: str) -> str:
    """Extract parsed hostname for audit purposes."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def frsc_match_key(url: str) -> str:
    """Match same FRSC URL across scheme/www/root-slash/fragment differences.

    Meaningful path and query remain part of the key. This deliberately avoids
    excluding every URL on a shared domain or hosted platform.
    """
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    port = parsed.port
    if port and not ((parsed.scheme.lower() == "http" and port == 80) or
                     (parsed.scheme.lower() == "https" and port == 443)):
        host = f"{host}:{port}"
    return f"{host}{path}?{parsed.query}"


def read_labelled_source(config: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load a labelled CSV, preserving source row numbers and invalid rows."""
    path = config["file"]
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")
    frame = pd.read_csv(path, usecols=[config["url_col"], config["label_col"]], low_memory=False)
    frame.columns = ["raw_url", "raw_label"]
    frame["source_row"] = range(2, len(frame) + 2)  # Account for CSV header row.
    frame["label"] = frame["raw_label"].map(lambda v: normalize_label(v, config["mapping"]))
    normalized = frame["raw_url"].map(normalize_url_with_reason)
    frame["url"] = normalized.map(lambda pair: pair[0])
    frame["invalid_reason"] = normalized.map(lambda pair: pair[1])
    invalid_mask = frame["url"].isna()
    unknown_mask = frame["label"].isna()
    invalid = frame.loc[invalid_mask].copy()
    invalid["source"] = config["source"]
    invalid = invalid[["source", "source_row", "raw_url", "raw_label", "invalid_reason"]]
    usable = frame.dropna(subset=["url", "label"])[["raw_url", "raw_label", "source_row", "url", "label"]].copy()
    usable["label"] = usable["label"].astype("int8")
    usable["source"] = config["source"]
    usable["submission_time"] = pd.NA
    report = {
        "source": config["source"], "file": path.name, "input_rows": len(frame),
        "usable_rows": len(usable), "invalid_url_rows": int(invalid_mask.sum()),
        "unknown_or_missing_label_rows": int(unknown_mask.sum()),
        "unknown_label_values": sorted(set(frame.loc[unknown_mask, "raw_label"].dropna().astype(str).str.strip())),
        "phishing_rows": int((usable["label"] == 1).sum()),
        "legitimate_rows": int((usable["label"] == 0).sum()),
    }
    return usable, report, invalid


def read_phishtank() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Import PhishTank rows marked verified=yes and online=yes."""
    if not PHISHTANK_FILE.exists():
        raise FileNotFoundError(f"PhishTank file not found: {PHISHTANK_FILE}")
    required = ["url", "verified", "online", "submission_time"]
    frame = pd.read_csv(PHISHTANK_FILE, usecols=required, low_memory=False)
    status = (frame["verified"].astype(str).str.strip().str.lower().eq("yes") &
              frame["online"].astype(str).str.strip().str.lower().eq("yes"))
    filtered = frame.loc[status].copy()
    filtered["raw_url"] = filtered["url"].astype(str)
    filtered["source_row"] = filtered.index + 2
    normalized = filtered["url"].map(normalize_url_with_reason)
    filtered["url"] = normalized.map(lambda pair: pair[0])
    filtered["invalid_reason"] = normalized.map(lambda pair: pair[1])
    invalid = filtered.loc[filtered["url"].isna()].copy()
    invalid["source"] = "phishtank_dataset_10"
    invalid["raw_label"] = "verified_phishing_feed_entry"
    invalid = invalid[["source", "source_row", "raw_url", "raw_label", "invalid_reason"]]
    usable = filtered.dropna(subset=["url"])[["raw_url", "source_row", "url", "submission_time"]].copy()
    usable["raw_label"] = "verified_phishing_feed_entry"
    usable["label"] = 1
    usable["source"] = "phishtank_dataset_10"
    usable = usable[["raw_url", "raw_label", "source_row", "url", "label", "source", "submission_time"]]
    report = {
        "source": "phishtank_dataset_10", "file": PHISHTANK_FILE.name,
        "input_rows": len(frame), "verified_and_online_rows": int(status.sum()),
        "usable_rows": len(usable), "invalid_url_rows_after_status_filter": int(filtered["url"].isna().sum()),
        "unknown_or_missing_label_rows": 0, "phishing_rows": len(usable), "legitimate_rows": 0,
    }
    return usable, report, invalid


def read_reserved_frsc_urls() -> tuple[set[str], pd.DataFrame]:
    """Load the FRSC case-study URLs and create an auditable match table."""
    if not FRSC_FILE.exists():
        raise FileNotFoundError(f"Required FRSC case-study file not found: {FRSC_FILE}")
    columns = pd.read_csv(FRSC_FILE, nrows=0).columns
    url_col = next((col for col in columns if str(col).strip().lower() == "url"), None)
    if url_col is None:
        raise ValueError(f"No URL column found in {FRSC_FILE.name}; columns={list(columns)}")
    frame = pd.read_csv(FRSC_FILE, usecols=[url_col], low_memory=False).rename(columns={url_col: "raw_url"})
    frame["source_row"] = range(2, len(frame) + 2)
    normalized = frame["raw_url"].map(normalize_url_with_reason)
    frame["normalized_url"] = normalized.map(lambda pair: pair[0])
    frame["invalid_reason"] = normalized.map(lambda pair: pair[1])
    frame["match_key"] = frame["normalized_url"].map(lambda value: frsc_match_key(value) if value else None)
    frame["matched_training_candidate"] = False
    return set(frame["match_key"].dropna()), frame


def main() -> None:
    """Create audit reports and a deduplicated master URL dataset."""
    parts, reports, invalid_parts = [], [], []
    for config in SELECTED_SOURCES:
        part, report, invalid = read_labelled_source(config)
        parts.append(part); reports.append(report); invalid_parts.append(invalid)
        print(f"{report['source']}: {report['usable_rows']:,} usable rows")
    phish_part, phish_report, phish_invalid = read_phishtank()
    parts.append(phish_part); reports.append(phish_report); invalid_parts.append(phish_invalid)
    print(f"{phish_report['source']}: {phish_report['usable_rows']:,} usable rows")

    combined = pd.concat(parts, ignore_index=True)
    combined["hostname"] = combined["url"].map(hostname_of)
    combined["frsc_match_key"] = combined["url"].map(frsc_match_key)
    frsc_keys, frsc_audit = read_reserved_frsc_urls()
    before_frsc = len(combined)
    matched_mask = combined["frsc_match_key"].isin(frsc_keys)
    frsc_excluded_rows = int(matched_mask.sum())
    # Flag each FRSC URL that appears in the candidate sources before exclusion.
    candidate_keys = set(combined["frsc_match_key"].dropna())
    frsc_audit["matched_training_candidate"] = frsc_audit["match_key"].isin(candidate_keys)
    combined = combined.loc[~matched_mask].copy()
    frsc_audit.to_csv(OUTPUT_DIR / "frsc_exclusion_audit.csv", index=False)

    # Exclude URL keys with contradictory labels; never choose one label arbitrarily.
    label_counts = combined.groupby("url")["label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)
    conflicts = combined[combined["url"].isin(conflicting_urls)].sort_values(["url", "source"])
    conflicts.to_csv(OUTPUT_DIR / "conflicting_urls.csv", index=False)
    conflict_rows = int(combined["url"].isin(conflicting_urls).sum())
    clean = combined.loc[~combined["url"].isin(conflicting_urls)].copy()

    # Consolidate exact normalized duplicates while retaining provenance.
    clean = clean.sort_values(["url", "source", "submission_time"], na_position="last")
    master = clean.groupby(["url", "label"], as_index=False).agg(
        sources=("source", lambda values: ";".join(sorted(set(values)))),
        raw_urls=("raw_url", lambda values: " || ".join(dict.fromkeys(str(v) for v in values if pd.notna(v)))),
        source_rows=("source_row", lambda values: ";".join(str(int(v)) for v in values if pd.notna(v))),
        raw_labels=("raw_label", lambda values: ";".join(sorted(set(str(v) for v in values if pd.notna(v))))),
        submission_times=("submission_time", lambda values: ";".join(sorted(set(str(v) for v in values if pd.notna(v))))),
        hostname=("hostname", "first"),
    )

    # Mark IP literals for later analysis; do not use this as a feature yet.
    def is_ip(host: str) -> int:
        try:
            ipaddress.ip_address(host)
            return 1
        except ValueError:
            return 0
    master["is_ip_host"] = master["hostname"].map(is_ip).astype("int8")
    master = master.sort_values("url").reset_index(drop=True)
    master.to_csv(OUTPUT_DIR / "master_urls_v2.csv", index=False)

    # Keep rejected records for inspection rather than silently dropping them.
    invalid_report = pd.concat(invalid_parts, ignore_index=True)
    invalid_report.to_csv(OUTPUT_DIR / "invalid_urls_v2.csv", index=False)
    summary = {
        "project_label_convention": {"1": "phishing", "0": "legitimate"},
        "sources": reports,
        "combined_rows_before_frsc_exclusion": before_frsc,
        "reserved_frsc_unique_match_keys": len(frsc_keys),
        "frsc_case_study_rows": len(frsc_audit),
        "frsc_case_study_invalid_urls": int(frsc_audit["normalized_url"].isna().sum()),
        "frsc_urls_matched_to_candidate_pool": int(frsc_audit["matched_training_candidate"].sum()),
        "rows_excluded_by_frsc_match": frsc_excluded_rows,
        "invalid_url_rows_across_sources": len(invalid_report),
        "conflicting_unique_urls": len(conflicting_urls),
        "rows_excluded_due_to_conflicts": conflict_rows,
        "final_unique_url_rows": len(master),
        "final_phishing_rows": int((master["label"] == 1).sum()),
        "final_legitimate_rows": int((master["label"] == 0).sum()),
        "notes": [
            "Dataset 7 is held back until its ClassLabel mapping is confirmed.",
            "Tranco is not automatically labelled legitimate.",
            "Dataset 3 and feature-only dataset 8 are not included in this URL-level master.",
            "No model features or train/validation/test splits are generated.",
            "Rejected URLs are preserved in invalid_urls_v2.csv with rejection reasons.",
            "FRSC matching ignores scheme, leading www., fragment and trailing slash, but retains hostname, path and query.",
            "Only matching FRSC URLs are excluded; entire domains are not removed.",
            "PhishTank submission_time is not treated as collection time.",
            "Conflicting URL labels are excluded and listed in conflicting_urls.csv.",
            "Output is written to processed_v2_audited, preserving earlier processed_v2 output.",
        ],
    }
    with open(OUTPUT_DIR / "audit_summary_v2.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    pd.DataFrame(reports).to_csv(OUTPUT_DIR / "source_audit_v2.csv", index=False)

    print("\nPreprocessing v2 URL audit complete.")
    print(f"Final unique URLs: {len(master):,}")
    print(f"Phishing: {int((master['label'] == 1).sum()):,}")
    print(f"Legitimate: {int((master['label'] == 0).sum()):,}")
    print(f"FRSC URLs matched: {int(frsc_audit['matched_training_candidate'].sum()):,}")
    print(f"FRSC candidate rows excluded: {frsc_excluded_rows:,}")
    print(f"Invalid URL rows: {len(invalid_report):,}")
    print(f"Conflicting URLs excluded: {len(conflicting_urls):,}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
