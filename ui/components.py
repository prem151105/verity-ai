"""Escaped, responsive observatory diagram shared with the README capture."""
from html import escape

ROLES = {
    'planner': ('Plan', 'Define the research brief', '#d8ba79', 110, 60),
    'retriever': ('Retrieve', 'Gather independent sources', '#8dafc9', 350, 60),
    'analyst': ('Analyze', 'Compute & interpret', '#8dafc9', 110, 160),
    'skeptic': ('Challenge', 'Find gaps & counter-evidence', '#c5a2da', 350, 160),
    'writer': ('Write', 'Draft with source pointers', '#d8ba79', 110, 270),
    'verifier': ('Verify', 'Check evidence, not consensus', '#8fd5b8', 350, 270),
    'adjudicator': ('Adjudicate', 'Revise, stop or retain flags', '#c5a2da', 230, 370),
    'assembler': ('Publish', 'Assemble the research note', '#8fd5b8', 230, 465),
}


def observatory(events=None, demo=False):
    events = events or []
    statuses = {e['node']: e['kind'] for e in events}
    active = next((e['node'] for e in reversed(events) if statuses[e['node']] == 'started'), None)
    paths = [(110,80,350,80), (350,80,110,180), (110,180,350,180),
             (350,180,110,290), (110,290,350,290), (350,290,230,390), (230,390,230,485)]
    svg = '<svg viewBox="0 0 460 525" role="img" aria-label="Eight research roles with a bounded revision loop">'
    for x,y,xx,yy in paths:
        svg += f'<path d="M{x} {y} L{xx} {yy}" stroke="#343d3c" stroke-width="1.4" fill="none"/>'
    svg += '<path d="M155 390 Q18 390 35 290" stroke="#9d8654" stroke-dasharray="4 5" fill="none"/><text x="13" y="356" fill="#c8b488" font-size="10">revise</text>'
    for name,(label,desc,color,x,y) in ROLES.items():
        kind = statuses.get(name, 'queued')
        stroke = color if kind in ('started','completed') else '#3b4343'
        symbol = '●' if kind == 'started' else '✓' if kind == 'completed' else '×' if kind == 'failed' else '·'
        svg += f'<g class="{"active" if name == active else ""}"><title>{escape(desc)} — {kind}</title><rect x="{x-83}" y="{y}" width="166" height="43" rx="8" fill="#182021" stroke="{stroke}" stroke-width="{2 if name == active else 1}"/><text x="{x-66}" y="{y+26}" fill="{color}" font-size="13">{symbol}</text><text x="{x+4}" y="{y+26}" text-anchor="middle" fill="#e9ece7" font-size="13" font-weight="600">{label}</text></g>'
    svg += '</svg>'
    rows = ''
    for e in events[-10:]:
        color = ROLES.get(e['node'], ('','','#a6b1ac'))[2]
        desc = e.get('detail') or e['kind']
        rows += f'<div class="log-row"><span class="seq">{e["sequence"]:02}</span><b style="color:{color}">{escape(e["node"])}</b><span>{escape(desc)}</span></div>'
    if not rows:
        rows = '<div class="empty"><span class="cross">+</span><h3>A question starts the journey.</h3><p>Launch a research run or explore the synthetic case.<br>Agent handoffs will appear here as they happen.</p></div>'
    mode = 'SYNTHETIC DEMO · SCRIPTED ROLES / REAL LOCAL CHECKS' if demo else 'RESEARCH SESSION · OBSERVED AGENT EVENTS'
    return f'''<!doctype html><html><head><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#101515;color:#e3e9e4;font:13px system-ui,sans-serif}}
    .shell{{border:1px solid #303b37;border-radius:14px;overflow:hidden;background:linear-gradient(115deg,#141c19,#101515)}}
    .bar{{display:flex;justify-content:space-between;padding:17px 22px;border-bottom:1px solid #303b37;font:10px monospace;letter-spacing:1.3px;color:#b9c9bd}}
    .dot{{color:#a5deba}}.grid{{display:grid;grid-template-columns:1.1fr 1fr}}.logs{{padding:25px 20px;border-right:1px solid #303b37;height:500px;overflow-y:auto}}
    .log-row{{display:grid;grid-template-columns:20px 90px 1fr;gap:8px;padding:10px 0;font:11px/1.6 monospace;border-bottom:1px solid #202b25}}
    .seq{{color:#718277}}.log-row span:last-child{{color:#c2cbc5}}.map{{padding:0 18px;background-image:radial-gradient(#344039 0.65px,transparent 0.65px);background-size:18px 18px}}
    svg{{width:100%;max-height:500px}}.empty{{padding:90px 10px;color:#9baa9f}}.empty h3{{font:22px Georgia;color:#e7ece5}}.empty p{{line-height:1.8}}.cross{{font:45px Georgia;color:#b7d9a4}}
    .active{{filter:drop-shadow(0 0 7px #8fd5b844)}}.foot{{border-top:1px solid #303b37;padding:12px 22px;font:10px monospace;color:#9baa9f}}
    @media(max-width:640px){{.grid{{grid-template-columns:1fr}}.logs{{height:280px;border-right:0;border-bottom:1px solid #303b37}}.empty{{padding:10px}}.map{{max-height:420px}}svg{{height:410px}}.bar{{font-size:8px}}}}
    </style></head><body><div class="shell"><div class="bar"><span><span class="dot">●</span> {mode}</span><span>{len(events):02} EVENTS</span></div><div class="grid"><div class="logs">{rows}</div><div class="map">{svg}</div></div><div class="foot">{'ACTIVE / ' + escape(active.upper()) if active else 'EVIDENCE BEFORE CONCLUSIONS'} &nbsp; · &nbsp; bounded revisions &nbsp; · &nbsp; independent verification</div></div></body></html>'''
