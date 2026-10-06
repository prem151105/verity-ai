"""Narrow financial field checks against independently retained source statements.

Only a single observed field/value assertion qualifies. Predictions, comparisons,
negation, extra quantities and unsupported qualifiers stay with semantic review.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

SCALES = {'trillion': 10**12, 'tn': 10**12, 't': 10**12,
          'billion': 10**9, 'bn': 10**9, 'b': 10**9,
          'million': 10**6, 'm': 10**6, 'thousand': 10**3, 'k': 10**3}
NUMBER = re.compile(r'(?<!\w)([-+]?\d[\d,]*(?:\.\d+)?)\s*(%|trillion|billion|million|thousand|tn|bn|[TBMK]\b)?', re.I)


@dataclass(frozen=True)
class Quantity:
    value: Decimal
    unit: str
    step: Decimal
    rounded: bool


def quantities(text):
    result = []
    for amount, suffix in NUMBER.findall(text):
        raw = Decimal(amount.replace(',', ''))
        scale = Decimal(SCALES.get(suffix.lower(), 1))
        result.append(Quantity(raw * scale, 'percent' if suffix == '%' else 'number',
                               Decimal(10) ** raw.as_tuple().exponent * scale,
                               '.' in amount or suffix.lower() in SCALES))
    return result


def compatible(claim, evidence):
    if claim.unit != evidence.unit:
        return False
    if claim.value == evidence.value:
        return True
    # Only the precision actually displayed in the claim can be rounded.
    return claim.rounded and abs(claim.value - evidence.value) < claim.step / 2


def numeric_compatible(claim, evidence):
    return all(any(compatible(c, e) for e in quantities(evidence)) for c in quantities(claim))


# More specific aliases are tested first, so revenue growth cannot match revenue.
FIELDS = {
    'revenue_yoy_growth': (r'(?:year.over.year\s+revenue\s+(?:growth|increase)|revenue\s+(?:year.over.year|yoy)\s+growth|revenue\s+growth)',
                           r'revenue(?:[_ ]yoy[_ ]growth(?:_pct)?|\s+year.over.year\s+growth)'),
    'net_income_yoy_growth': (r'(?:net income\s+(?:year.over.year|yoy)\s+growth)', r'net[_ ]income[_ ]yoy[_ ]growth(?:_pct)?'),
    'net_margin': (r'net\s+(?:profit\s+)?margin', r'net[_ ]margin(?:_pct)?'),
    'gross_margin': (r'gross\s+margin', r'gross[_ ]margin(?:_pct)?'),
    'operating_margin': (r'operating\s+margin', r'operating[_ ]margin(?:_pct)?'),
    'market_cap': (r'market\s+(?:capitalization|cap)\b', r'market\s+(?:capitalization|cap)'),
    'forward_pe': (r'forward\s+p\s*/\s*e', r'forward\s+p\s*/\s*e'),
    'trailing_pe': (r'trailing\s+p\s*/\s*e|p\s*/\s*e(?:\s+ratio)?\s*\(trailing\)', r'p\s*/\s*e(?:\s+ratio)?\s*\(trailing\)|trailing\s+p\s*/\s*e'),
    'price_to_book': (r'price[ /-](?:to[ /-])?book', r'price[ /-](?:to[ /-])?book'),
    'debt_to_equity': (r'debt[ /-](?:to[ /-])?equity', r'debt[_ /-](?:to[_ /-])?equity'),
    'current_price': (r'(?:current|share|stock)\s+price', r'current\s+price'),
}
UNSUPPORTED = re.compile(r'\b(?:not|no|never|above|below|less|more|highest|lowest|leading|unprecedented|'
                         r'will|would|should|may|could|suggest\w*|anticipat\w*|expect\w*|forecast\w*|'
                         r'undervalued|overvalued|because|indicat\w*|prove\w*|fail\w*|incorrect|false|'
                         r'prior|previous|historical\w*|future|next|last|quarter.over.quarter)\b', re.I)


def field_match(claim, evidence, source, entity_aliases=()):
    """Return a support reason only when a single financial observation matches."""
    clean = re.sub(r'[*_`]', ' ', claim).strip()
    if UNSUPPORTED.search(clean) or re.search(r"n['’]t\b|[<>]", clean, re.I) or len(quantities(clean)) != 1:
        return None
    # Avoid asserting a different named ticker against the cited company.
    source_ticker = source.rsplit('—', 1)[-1].strip() if '—' in source else ''
    mentioned = set(re.findall(r'\b[A-Z]{2,15}\b', clean)) - {'USD', 'YOY', 'CORP', 'INC'}
    allowed = {source_ticker} if source_ticker.isupper() else set()
    for alias in entity_aliases:
        allowed.update(re.findall(r'\b[A-Z]{2,15}\b', alias.upper()))
    if mentioned - allowed:
        return None
    matched = [key for key,(alias, _) in FIELDS.items() if re.search(alias, clean, re.I)]
    if len(matched) != 1:
        return None
    key = matched[0]
    if 'growth' not in key and re.search(r'\b(?:rose|grew|increas\w*|decreas\w*|improv\w*|expand\w*|surge\w*|jump\w*)\b', clean, re.I):
        return None
    value_pattern = FIELDS[key][1]
    # Evidence must state a labeled value, not merely mention the concept.
    pattern = rf'^\s*(?:{value_pattern})\s*(?::|was|is|=)\s*\$?([-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|trillion|billion|million|thousand|tn|bn|[TBMK]\b)?)'
    match = re.search(pattern, evidence, re.I)
    if not match or not numeric_compatible(clean, match.group(1)):
        return None
    if re.search(r'\bexactly\b', clean, re.I) and quantities(clean)[0].value != quantities(match.group(1))[0].value:
        return None
    value = quantities(clean)[0].value
    if re.search(r'\b(?:declin\w*|decreas\w*|fell|down)\b', clean, re.I) and value >= 0:
        return None
    if re.search(r'\b(?:increas\w*|grew|growth)\b', clean, re.I) and value < 0:
        return None
    return f'Single {key.replace("_", " ")} observation matches the source field at the displayed precision'
