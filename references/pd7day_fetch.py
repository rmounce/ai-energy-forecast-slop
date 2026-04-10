#!/usr/bin/env python3
import csv
import json
import re
import sys
import urllib.request
import zipfile
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = 'https://www.nemweb.com.au/REPORTS/CURRENT/PD7Day/'
FILE_PATTERN = re.compile(r'PUBLIC_PD7DAY_.*\.(ZIP|CSV)$', re.IGNORECASE)


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def parse_dt(s: str) -> datetime:
    return datetime.strptime(s, '%Y/%m/%d %H:%M:%S')


def to_iso_local(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%dT%H:%M:%S')


def list_remote_pd7day_files(base_url: str = BASE_URL):
    with urllib.request.urlopen(base_url) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
    parser = LinkExtractor()
    parser.feed(html)
    files = []
    for href in parser.links:
        name = href.split('/')[-1]
        if FILE_PATTERN.search(name):
            files.append({'name': name, 'url': urljoin(base_url, href)})
    return files


def newest_remote_pd7day(base_url: str = BASE_URL):
    files = list_remote_pd7day_files(base_url)
    if not files:
        raise FileNotFoundError('No PUBLIC_PD7DAY ZIP/CSV files found at NEMWeb')
    return sorted(files, key=lambda x: x['name'])[-1]


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open('wb') as f:
        f.write(resp.read())
    return dest


def extract_zip_to_csv(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = [m for m in zf.namelist() if m.upper().endswith('.CSV')]
        if not members:
            raise FileNotFoundError(f'No CSV found inside {zip_path}')
        member = sorted(members)[0]
        out_path = zip_path.with_suffix('.CSV')
        with zf.open(member) as src, out_path.open('wb') as dst:
            dst.write(src.read())
    return out_path


def find_local_newest(search_dir: Path):
    candidates = sorted(list(search_dir.glob('PUBLIC_PD7DAY*.CSV')) + list(search_dir.glob('PUBLIC_PD7DAY*.zip')) + list(search_dir.glob('PUBLIC_PD7DAY*.ZIP')))
    return candidates[-1] if candidates else None


def resolve_input_csv(arg_path: str | None):
    if arg_path:
        p = Path(arg_path)
        if not p.exists():
            raise FileNotFoundError(f'Input file not found: {arg_path}')
        if p.suffix.upper() == '.ZIP':
            return extract_zip_to_csv(p)
        return p

    try:
        newest = newest_remote_pd7day()
        downloaded = download_file(newest['url'], Path(newest['name']))
        return extract_zip_to_csv(downloaded) if downloaded.suffix.upper() == '.ZIP' else downloaded
    except Exception:
        local = find_local_newest(Path('.'))
        if local:
            return extract_zip_to_csv(local) if local.suffix.upper() == '.ZIP' else local
        raise


def load_pd7day_prices(csv_path: Path, region: str):
    prices = []
    run_dt = None

    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] != 'D':
                continue
            if len(row) < 20:
                continue
            if row[1] != 'PD7DAY' or row[2] != 'PRICESOLUTION':
                continue
            if row[7] != region:
                continue

            this_run = parse_dt(row[4])
            if run_dt is None:
                run_dt = this_run

            prices.append({
                'time': to_iso_local(parse_dt(row[6])),
                'value': round(float(row[8]) / 1000.0, 6)
            })

    prices.sort(key=lambda x: x['time'])
    return run_dt, prices


def average_price(prices_slice):
    if not prices_slice:
        return None
    return round(sum(p['value'] for p in prices_slice) / len(prices_slice), 6)


def find_cheapest_window(prices, hours=2, interval_minutes=30):
    window_points = int(hours * 60 / interval_minutes)
    if len(prices) < window_points:
        return None

    best = None
    for i in range(len(prices) - window_points + 1):
        window = prices[i:i + window_points]
        avg = average_price(window)
        if best is None or avg < best['avg_value']:
            best = {
                'start': window[0]['time'],
                'end': window[-1]['time'],
                'avg_value': avg,
                'points': len(window)
            }
    return best


def min_max_for_horizon(prices, hours=24):
    if not prices:
        return None, None
    subset = prices[: int(hours * 2)]
    if not subset:
        return None, None
    vals = [p['value'] for p in subset]
    return round(min(vals), 6), round(max(vals), 6)


def build_payload(csv_path: Path, region: str):
    run_dt, prices = load_pd7day_prices(csv_path, region)
    if not prices:
        raise ValueError(f'No PRICESOLUTION rows found for region {region}')

    current_price = prices[0]['value']
    next_price = prices[1]['value'] if len(prices) > 1 else None
    min_24h, max_24h = min_max_for_horizon(prices, hours=24)
    cheapest_2h = find_cheapest_window(prices, hours=2, interval_minutes=30)

    payload = {
        'source_file': str(csv_path),
        'region': region,
        'forecast_generated_at': to_iso_local(run_dt) if run_dt else None,
        'unit': '$/kWh',
        'interval_minutes': 30,
        'current_value': round(current_price, 6),
        'next_value': round(next_price, 6) if next_price is not None else None,
        'min_24h_value': min_24h,
        'max_24h_value': max_24h,
        'cheapest_2h_window': cheapest_2h,
        'forecast': prices,
    }
    return payload


def build_home_assistant_payload(payload: dict, entity_id: str):
    attributes = {
        'friendly_name': f"{payload['region']} PD7DAY Forecast",
        'icon': 'mdi:transmission-tower',
        'unit_of_measurement': payload['unit'],
        'region': payload['region'],
        'forecast_generated_at': payload['forecast_generated_at'],
        'interval_minutes': payload['interval_minutes'],
        'next_value': payload['next_value'],
        'min_24h_value': payload['min_24h_value'],
        'max_24h_value': payload['max_24h_value'],
        'cheapest_2h_window': payload['cheapest_2h_window'],
        'forecast': payload['forecast'],
        'source_file': payload['source_file'],
    }
    return {
        'entity_id': entity_id,
        'state': payload['current_value'],
        'attributes': attributes,
    }


def main():
    args = sys.argv[1:]
    output_mode = 'ha'

    if '--mode' in args:
        i = args.index('--mode')
        output_mode = args[i + 1]
        del args[i:i + 2]

    entity_id = 'sensor.qld1_pd7day_forecast'
    if '--entity-id' in args:
        i = args.index('--entity-id')
        entity_id = args[i + 1]
        del args[i:i + 2]

    if len(args) >= 1 and args[0].lower().endswith(('.csv', '.zip')):
        csv_arg = args[0]
        region = args[1] if len(args) >= 2 else 'QLD1'
        output_json = Path(args[2]) if len(args) >= 3 else Path(f'pd7day_{region.lower()}_{output_mode}.json')
    else:
        csv_arg = None
        region = args[0] if len(args) >= 1 else 'QLD1'
        output_json = Path(args[1]) if len(args) >= 2 else Path(f'pd7day_{region.lower()}_{output_mode}.json')

    csv_path = resolve_input_csv(csv_arg)
    payload = build_payload(csv_path, region)

    if output_mode == 'raw':
        final_payload = payload
    else:
        final_payload = build_home_assistant_payload(payload, entity_id)

    rendered = json.dumps(final_payload, indent=2)
    output_json.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
