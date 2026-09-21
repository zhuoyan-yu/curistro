"""Text-safe session exports; content is never interpreted as HTML."""
from pathlib import Path
from fontTools.ttLib import TTFont
from fpdf import FPDF


class ExportError(Exception):
    pass


def sections(node, mode):
    result = [(1, f'Curistro Session - {node.title}'), (0, f'Category: {node.category}'),
              (0, f'Mode: {mode}' + (' (scripted sample, not an assessment)' if mode == 'demo' else '')),
              (0, f"AI student checks: {node.report['correct']} / {node.report['total']}"),
              (0, 'This score describes AI student answers, not a human exam score or validated learning gains.'),
              (2, 'Your explanation'), (0, node.content), (2, 'Conversation')]
    for m in node.conversation:
        result.append((0, ('You (teacher): ' if m['role'] == 'user' else 'AI student: ') + m['message']))
    result.append((2, 'Diagnostic questions and AI answers'))
    for q in node.report['details']:
        result.extend([(3, q['dimension']), (0, q['question'])])
        result.extend((0, f'{chr(65+i)}. {option}') for i, option in enumerate(q['options']))
        result.extend([(0, f"Answer key: {q['answer']}; AI choice: {q['choice'] or 'Abstained'}"), (0, q['rationale'])])
    if node.recap:
        result.extend([(2, 'Follow-up recap'), (0, node.recap)])
    return result


def markdown_export(node, mode):
    def escape(text):
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        for char in ('\\', '`', '*', '_', '[', ']', '#'):
            text = text.replace(char, '\\' + char)
        return text
    return '\n\n'.join(('#' * level + ' ' if level else '') + escape(text)
                        for level, text in sections(node, mode)) + '\n'


def pdf_export(node, mode, root, configured_font=''):
    font_path = Path(configured_font).expanduser() if configured_font else root / 'static/fonts/DejaVuSans.ttf'
    if not font_path.is_absolute():
        font_path = root / font_path
    if not font_path.is_file():
        raise ExportError('PDF font not found. Configure CURISTRO_PDF_FONT or download Markdown.')
    content = sections(node, mode)
    try:
        with TTFont(font_path) as font:
            supported = font.getBestCmap()
        unsupported = {c for _, text in content for c in text if not c.isspace() and ord(c) not in supported}
        if unsupported:
            raise ExportError('The selected PDF font cannot display some characters. Download Markdown, or configure CURISTRO_PDF_FONT with a font supporting your language.')
        pdf = FPDF()
        pdf.set_margins(18, 18, 18)
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_font('Session', fname=str(font_path))
        pdf.add_page()
        for level, text in content:
            pdf.set_font('Session', size={0: 10, 1: 18, 2: 14, 3: 11}[level])
            if level:
                pdf.ln(3)
            pdf.multi_cell(0, 6 if not level else 8, text.replace('\t', '    '), new_x='LMARGIN', new_y='NEXT', align='L')
            pdf.ln(2)
        return bytes(pdf.output())
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError('PDF export failed. Download Markdown or check the configured PDF font.') from exc
