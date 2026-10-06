"""Offline synthetic case. Only verification is real; other steps are fixtures."""
from agents.runtime import ResearchRuntime, initial_state, adjudicate
from research.cascade import Cascade

SOURCE = 'ExampleCo annual filing (synthetic)'
DOCUMENT = 'Revenue was $120 million. Operating income was $24 million. Customer concentration remains a material risk.'


def demo_runtime():
    def planner(s):
        return {**s, 'company_name': 'ExampleCo / synthetic case', 'demo': True,
                'task_list': ['Read the supplied example filing', 'Check the revenue claim', 'Retain unresolved issues']}

    def retriever(s):
        return {**s, 'filing_texts': [{'source': SOURCE, 'text': DOCUMENT}]}

    def analyst(s):
        return {**s, 'analyst_summary': 'Revenue and operating income are available; no market valuation data supplied.'}

    def skeptic(s):
        return {**s, 'skeptic_review': 'Customer concentration is disclosed. The initial revenue claim needs a numeric check.'}

    def writer(s):
        claim = 'Revenue was $900 million.' if not s.get('verifier_iteration') else 'Revenue was $120 million.'
        return {**s, 'draft_report': f'## Executive Summary\n{claim} [[CITE: {SOURCE} | Revenue was $120 million.]]\n\n'
                '## Key Risks\nCustomer concentration remains a material risk. '
                f'[[CITE: {SOURCE} | Customer concentration remains a material risk.]]',
                'citations': [{'claim': claim, 'source': SOURCE, 'passage': 'Revenue was $120 million.'},
                              {'claim': 'Customer concentration remains a material risk.', 'source': SOURCE,
                               'passage': 'Customer concentration remains a material risk.'}]}

    def verifier(s):
        result = Cascade().verify(s['citations'], s['filing_texts'])
        citations = [{**c, 'verified': v['supported'], 'verification_route': v['route'],
                      'verifier_reasoning': v['reasoning'], 'evidence': v.get('evidence'),
                      'confidence': v.get('confidence', 0)} for c, v in zip(s['citations'], result['verdicts'])]
        issues = [c['claim'] for c in citations if not c['verified']]
        return {**s, 'citations': citations, 'unverified_claims': issues,
                'verification_metrics': result['metrics'], 'verifier_iteration': s['verifier_iteration'] + 1,
                'verifier_feedback': 'Correct the revenue using the original filing.' if issues else ''}

    def assembler(s):
        from agents.assembler import _render_citations
        return {**s, 'final_report': '# ExampleCo research note\n\n'
                '> Synthetic demonstration, not company data. Planning, analysis, review and writing are scripted; '
                'the local verification cascade actually executes.\n\n' + _render_citations(s['draft_report']) +
                '\n\n## Evidence\n' + '\n'.join(f'[^'+str(i)+']: '+c['source']+' — '+c['passage']
                                                  for i,c in enumerate(s['citations'], 1))}
    return ResearchRuntime(dict(planner=planner, retriever=retriever, analyst=analyst,
                                skeptic=skeptic, writer=writer, verifier=verifier,
                                adjudicator=adjudicate, assembler=assembler))


def run_demo():
    return demo_runtime().invoke(initial_state('DEMO'))
