"""Capture the actual UI and render the synthetic event sequence as a README GIF.

Development only: pip install playwright pillow; playwright install chromium.
Start Streamlit first. This captures fixtures, not a live financial research run.
"""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.demo import demo_runtime
from agents.runtime import initial_state
from ui.components import observatory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://localhost:8501')
    parser.add_argument('--channel', default=None, help='Optional installed browser, such as msedge')
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright
    from PIL import Image
    output = ROOT / 'assets' / 'readme'
    states = [next(iter(p.values())) for p in demo_runtime().stream(initial_state('DEMO'))]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel=args.channel)
        page = browser.new_page(viewport={'width': 1440, 'height': 1160}, device_scale_factor=1)
        page.goto(args.url, wait_until='domcontentloaded')
        page.get_by_text('Follow the evidence.', exact=False).wait_for(timeout=30000)
        page.wait_for_timeout(1800)
        page.screenshot(path=str(output / 'observatory.png'), full_page=True)
        page.get_by_role('button', name='Explore a sample run').click()
        page.get_by_role('tab', name='Evidence ledger').wait_for(timeout=30000)
        page.get_by_role('tab', name='Evidence ledger').click()
        page.get_by_text('01 · SUPPORTED', exact=False).wait_for()
        page.wait_for_timeout(800)
        page.screenshot(path=str(output / 'evidence-ledger.png'), full_page=True)
        print('Desktop screenshot and synthetic sample interaction passed.')
        page.set_viewport_size({'width':390,'height':844})
        page.evaluate("document.querySelector('[data-testid=stMain]').scrollTop = 0")
        page.wait_for_timeout(500)
        page.screenshot(path=str(output / 'mobile.png'), full_page=True)
        overflow = page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
        print('Mobile horizontal overflow:', overflow)
        page.close()
        graph = browser.new_page(viewport={'width': 1060, 'height': 620}, device_scale_factor=1)
        frames = []
        with tempfile.TemporaryDirectory() as tmp:
            final_events = states[-1]['events']
            for i in range(len(final_events)):
                graph.set_content(observatory(final_events[:i+1], demo=True))
                graph.screenshot(path=str(Path(tmp) / f'{i}.png'))
                with Image.open(Path(tmp) / f'{i}.png') as frame:
                    frames.append(frame.convert('RGB'))
            frames[0].save(output / 'research-session.gif', save_all=True, append_images=frames[1:],
                           duration=[850] * (len(frames)-1) + [2300], loop=0, optimize=True)
        browser.close()
    print('Wrote research-session.gif from actual demo runtime events.')


if __name__ == '__main__':
    main()
