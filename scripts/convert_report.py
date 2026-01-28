
import markdown
import os

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Premium Academic HTML Template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bridging the Structure-Gap: An Empirical Study on Layout-Aware Parsing for Financial RAG</title>
    <style>
        /* === CSS Reset & Base === */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        /* === Typography === */
        :root {
            --font-serif: 'Georgia', 'Times New Roman', serif;
            --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, Helvetica, Arial, sans-serif;
            --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            
            --color-text: #1a202c;
            --color-text-muted: #4a5568;
            --color-accent: #2b6cb0;
            --color-bg: #ffffff;
            --color-border: #e2e8f0;
            --color-code-bg: #f7fafc;
            --color-table-stripe: #f7fafc;
        }

        html { font-size: 16px; }

        body {
            font-family: var(--font-serif);
            line-height: 1.75;
            color: var(--color-text);
            background-color: var(--color-bg);
            max-width: 800px;
            margin: 0 auto;
            padding: 60px 40px;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
        }

        /* === Headings === */
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-sans);
            font-weight: 700;
            color: var(--color-text);
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            line-height: 1.3;
        }

        h1 {
            font-size: 1.8rem;
            text-align: center;
            border-bottom: none;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }

        h2 {
            font-size: 1.35rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid var(--color-border);
            margin-top: 3rem;
        }

        h3 {
            font-size: 1.15rem;
            color: var(--color-text-muted);
        }

        h4 { font-size: 1rem; }

        /* === Body Text === */
        p {
            margin-bottom: 1.25rem;
            text-align: justify;
            hyphens: auto;
        }

        /* === Lists === */
        ul, ol {
            margin-bottom: 1.25rem;
            padding-left: 1.75rem;
        }

        li {
            margin-bottom: 0.5rem;
        }

        li > ul, li > ol {
            margin-top: 0.5rem;
            margin-bottom: 0;
        }

        /* === Links === */
        a {
            color: var(--color-accent);
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.2s ease;
        }

        a:hover {
            border-bottom-color: var(--color-accent);
        }

        /* === Blockquotes === */
        blockquote {
            border-left: 4px solid var(--color-accent);
            margin: 1.5rem 0;
            padding: 0.75rem 1.25rem;
            background-color: var(--color-code-bg);
            font-style: italic;
            color: var(--color-text-muted);
        }

        blockquote p:last-child { margin-bottom: 0; }

        /* === Code === */
        code {
            font-family: var(--font-mono);
            font-size: 0.85em;
            background-color: var(--color-code-bg);
            padding: 0.15em 0.4em;
            border-radius: 4px;
            border: 1px solid var(--color-border);
        }

        pre {
            font-family: var(--font-mono);
            font-size: 0.85rem;
            background-color: var(--color-code-bg);
            border: 1px solid var(--color-border);
            border-radius: 6px;
            padding: 1rem 1.25rem;
            overflow-x: auto;
            margin: 1.5rem 0;
            line-height: 1.5;
        }

        pre code {
            background: none;
            padding: 0;
            border: none;
            font-size: inherit;
        }

        /* === Tables === */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.75rem 0;
            font-size: 0.9rem;
        }

        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border: 1px solid var(--color-border);
        }

        th {
            font-family: var(--font-sans);
            font-weight: 600;
            background-color: var(--color-table-stripe);
            color: var(--color-text);
        }

        tbody tr:nth-child(even) {
            background-color: var(--color-table-stripe);
        }

        /* Table caption styling */
        table + p > em:first-child,
        table + p > strong:first-child {
            display: block;
            text-align: center;
            font-size: 0.85rem;
            color: var(--color-text-muted);
            margin-top: -1rem;
            margin-bottom: 1.5rem;
        }

        /* === Images === */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5rem auto;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        p img {
            margin: 0 auto;
        }

        /* === Horizontal Rules === */
        hr {
            border: none;
            border-top: 1px solid var(--color-border);
            margin: 3rem 0;
        }

        /* === Strong/Em === */
        strong { font-weight: 700; }
        em { font-style: italic; }

        /* === Figure/Caption (for image pairs) === */
        p[align="center"] {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 1rem;
            margin: 2rem 0;
        }

        p[align="center"] img {
            margin: 0;
            max-width: 48%;
        }

        /* === Print Styles === */
        @media print {
            html { font-size: 11pt; }

            body {
                padding: 0;
                max-width: 100%;
                line-height: 1.6;
            }

            h1, h2, h3, h4, h5, h6 {
                page-break-after: avoid;
            }

            table, img, blockquote, pre {
                page-break-inside: avoid;
            }

            a {
                color: var(--color-text);
                border-bottom: none;
            }

            a[href^="http"]::after {
                content: none; /* Don't print URLs */
            }

            blockquote {
                background-color: transparent;
                border-left-color: #333;
            }

            th {
                background-color: #eee !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }

            tbody tr:nth-child(even) {
                background-color: transparent !important;
            }
        }

        /* === First h1 special styling (title) === */
        body > h1:first-of-type {
            margin-top: 0;
            padding-bottom: 1.5rem;
            margin-bottom: 1.5rem;
        }

        /* === Abstract styling === */
        body > h2#abstract + p {
            font-size: 0.95rem;
        }
    </style>
</head>
<body>
{{content}}
</body>
</html>"""

    # Extensions for tables, fenced code, TOC, etc.
    html_content = markdown.markdown(
        text,
        extensions=['tables', 'fenced_code', 'toc', 'nl2br']
    )
    
    final_html = html_template.replace('{{content}}', html_content)

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"✓ Successfully generated: {html_path}")
    print(f"  Open in browser and use 'Print → Save as PDF' for best results.")

if __name__ == "__main__":
    md_file = r"c:\Users\22304\Desktop\structure-aware-rag-study\report\README.md"
    html_file = r"c:\Users\22304\Desktop\structure-aware-rag-study\report\Technical_Report_Structure_Aware_RAG.html"
    convert_md_to_html(md_file, html_file)
