from research.cascade import Cascade, Policy

MARKET = 'Market Data (yfinance) — NVDA'
DOCS = [
    {'source': MARKET, 'text': 'Market Cap: $5,768,717,795,328\nP/E Ratio (trailing): 30.202276\nForward P/E: 15.12099'},
    {'source': 'Financial Ratios Computation', 'text': 'Revenue yoy growth was 65.47%. Calculation source: Revenue YoY Growth (10-K filings).\nNet margin was 62.03%.'},
]


def test_saved_live_financial_observations_and_rounding():
    claims = [
        {'source': MARKET, 'claim': 'The company’s market capitalization stands at approximately $5.77 trillion'},
        {'source': 'Financial Ratios Computation', 'claim': 'Financial performance is characterized by a year-over-year revenue growth rate of 65.47%'},
        {'source': 'Financial Ratios Computation', 'claim': 'and a net margin of 62.03%'},
        {'source': MARKET, 'claim': 'While the stock trades at a trailing P/E of 30.20'},
    ]
    result = Cascade().verify(claims, DOCS)
    assert all(v['supported'] and v['route'] == 'financial_field' for v in result['verdicts'])
    assert result['metrics']['remote_attempts'] == 0
    assert result['verdicts'][2]['evidence']['text'] == 'Net margin was 62.03%.'


def test_wrong_values_entities_units_and_forecasts_fail_closed():
    claims = [
        {'source': MARKET, 'claim': 'The company’s market capitalization is $6.77 trillion'},
        {'source': MARKET, 'claim': 'AMD market capitalization is $5.77 trillion'},
        {'source': MARKET, 'claim': 'Trailing P/E was 30.20%'},
        {'source': MARKET, 'claim': "Trailing P/E wasn't 30.20"},
        {'source': MARKET, 'claim': 'Trailing P/E increased to 30.20'},
        {'source': MARKET, 'claim': 'Market capitalization is exactly $5.77 trillion'},
        {'source': MARKET, 'claim': 'The forward P/E of 15.12 suggests significant anticipated earnings expansion'},
        {'source': 'Financial Ratios Computation', 'claim': 'Revenue declined by 65.47%'},
    ]
    assert not any(v['supported'] for v in Cascade().verify(claims, DOCS)['verdicts'])


def test_batch_and_cache_preserve_budget_and_changed_evidence():
    calls = []
    def batch(items):
        calls.append(items)
        return {i: {'supported': True, 'confidence': .99, 'reasoning': 'Evidence supports the statement'}
                for i, *_ in items}
    cache = {}
    claims = [{'source':'source','claim':'Revenue expanded.'}, {'source':'source','claim':'Debt declined.'}]
    docs = [{'source':'source','text':'Revenue increased. Debt fell.'}]
    cascade = Cascade(Policy(remote_budget=2), remote_batch=batch, cache=cache)
    first = cascade.verify(claims, docs)
    second = cascade.verify(claims, docs)
    assert len(calls) == 1
    assert first['metrics']['remote_requests'] == 1
    assert first['metrics']['remote_attempts'] == 2
    assert second['metrics']['cache_hits'] == 2 and second['metrics']['remote_attempts'] == 0
    cascade.verify(claims, [{'source':'source','text':'Revenue declined. Debt increased.'}])
    assert len(calls) == 2
