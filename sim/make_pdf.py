"""把 docs/NO_mouse_brain_analysis.md 渲染为 PDF (results/NO_mouse_brain_report.pdf)。
依赖: pip install markdown ; Playwright + Chromium。"""
import os, pathlib, subprocess, markdown

root = pathlib.Path(__file__).resolve().parent.parent
md = (root / "docs/NO_mouse_brain_analysis.md").read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'WenQuanYi Zen Hei', sans-serif; font-size: 10.5pt; line-height: 1.6; color: #222; }
h1 { font-size: 20pt; color: #0d3b66; border-bottom: 3px solid #0d3b66; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #0d3b66; margin-top: 22px; border-left: 5px solid #1f6fd1; padding-left: 8px; }
h3 { font-size: 12pt; color: #1f4e79; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #b8c4d0; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #e8eff7; }
td:first-child, th:first-child { white-space: nowrap; }
strong { color: #0d3b66; }
tr:nth-child(even) td { background: #f7f9fc; }
pre { background: #f4f6f8; border: 1px solid #dde3ea; padding: 8px; font-size: 8.4pt; line-height: 1.35;
      font-family: 'WenQuanYi Zen Hei Mono', monospace; white-space: pre; page-break-inside: avoid; }
code { font-family: 'WenQuanYi Zen Hei Mono', monospace; font-size: 9pt; background: #f1f3f5; padding: 0 3px; }
pre code { background: none; padding: 0; }
blockquote { color: #555; border-left: 4px solid #ccc; margin: 0; padding: 2px 10px; background: #fafafa; }
img { display: block; max-width: 100%; margin: 10px auto; page-break-inside: avoid; }
p img + br { display: none; }
hr { border: none; border-top: 1px solid #ddd; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
tmp = root / "docs/_report.html"
tmp.write_text(html, encoding="utf-8")
out = root / "results/NO_mouse_brain_report.pdf"
js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://{tmp}', {{ waitUntil: 'networkidle' }});
  await p.pdf({{ path: '{out}', format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: {{ top: '16mm', bottom: '18mm', left: '14mm', right: '14mm' }} }});
  await b.close();
}})();
"""
subprocess.run(["node", "-e", js], check=True)
tmp.unlink()
print(out)
