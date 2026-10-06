"""Run an actual API research job and save its completed report. No fixtures."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def request(url, method='GET'):
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', nargs='?', default='NVDA')
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    if args.run_id:
        run_id = args.run_id
    else:
        code, started = request(f'{args.api}/research/{args.ticker}', 'POST')
        if code != 200:
            raise RuntimeError(f'Could not start research: {started}')
        run_id = started['run_id']
    print(f'Live run: {run_id}', flush=True)
    logs = ROOT / 'audit_logs'
    logs.mkdir(exist_ok=True)
    (logs / 'live-validation-run.txt').write_text(run_id)
    sequence = 0
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        code, payload = request(f'{args.api}/research/{run_id}/events?after={sequence}')
        if code != 200:
            raise RuntimeError(f'Events request failed: {payload}')
        for event in payload['events']:
            sequence = event['sequence']
            print(f"{sequence:02} {event['node']} / {event['kind']} / {event['detail']}", flush=True)
        if payload['status'] == 'completed':
            code, result = request(f'{args.api}/research/{run_id}')
            if code != 200 or not result.get('final_report'):
                raise RuntimeError(f'Completed report missing: {result}')
            report = ROOT / 'reports' / f'{args.ticker.upper()}_live_research.md'
            report.write_text(result['final_report'], encoding='utf-8')
            (logs / 'live-validation-result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(f"COMPLETED: {result['verified_citation_count']}/{result['citation_count']} cited claims supported; "
                  f"{len(result['unverified_claims'])} unresolved. Report: {report}", flush=True)
            return
        if payload['status'] in ('failed', 'cancelled', 'interrupted'):
            raise RuntimeError(f"Live run {payload['status']}: {payload.get('error')}")
        time.sleep(3)
    raise TimeoutError(f'Live run exceeded 20 minutes; inspect run {run_id} in the dashboard.')


if __name__ == '__main__':
    main()
