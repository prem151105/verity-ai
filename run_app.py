"""Start Verity's API and dashboard together using the project environment."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    project_python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if project_python.exists() and Path(sys.executable).resolve() != project_python.resolve():
        return subprocess.call([str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT)
    os.chdir(ROOT)
    from config import settings
    if not settings.gemini_api_key or settings.gemini_api_key.startswith('your_'):
        print('Add GEMINI_API_KEY to .env in the project folder. .env.example is a template.')
        return 1
    logs = ROOT / settings.audit_log_dir
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    handles = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        health = get_json('http://127.0.0.1:8000/health')
        if not health:
            log = open(logs / 'api-server.log', 'a', encoding='utf-8')
            handles.append(log)
            children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.main:app',
                                              '--host', '127.0.0.1', '--port', '8000'],
                                             stdout=log, stderr=log, cwd=ROOT, creationflags=flags))
            for _ in range(30):
                health = get_json('http://127.0.0.1:8000/health')
                if health:
                    break
                time.sleep(1)
        if not health or health.get('service') != 'Verity':
            print('The research API could not start. Check audit_logs/api-server.log or port 8000.')
            return 1
        log = open(logs / 'dashboard.log', 'a', encoding='utf-8')
        handles.append(log)
        children.append(subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'ui/app.py',
                                          '--server.address', '127.0.0.1', '--server.port', str(args.port),
                                          '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                                         stdout=log, stderr=log, cwd=ROOT, creationflags=flags))
        url = f'http://127.0.0.1:{args.port}'
        for _ in range(30):
            try:
                urllib.request.urlopen(url + '/_stcore/health', timeout=2).close()
                break
            except OSError:
                if children[-1].poll() is not None:
                    print('Dashboard failed to start. Check audit_logs/dashboard.log.')
                    return 1
                time.sleep(1)
        else:
            print('Dashboard startup timed out. Check audit_logs/dashboard.log.')
            return 1
        print(f'Verity is ready: {url}\nPress Ctrl+C here to stop the services started by this launcher.')
        if not args.no_browser:
            webbrowser.open(url)
        while all(child.poll() is None for child in children):
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopping Verity.')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
        for handle in handles:
            handle.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
