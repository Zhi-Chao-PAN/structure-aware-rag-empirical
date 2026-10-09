import pandas as pd
import json
import random
import os

def generate_synthetic_visfinqa(metadata_path, output_dir, num_queries=500):
    df = pd.read_parquet(metadata_path)
    
    # Templates for standard financial metrics found in 10-K/10-Q documents
    templates = [
        ("What was {company}'s total revenue in {year} {quarter}?", "Net sales", "revenue"),
        ("What was the net income for {company} in {year}?", "Net income", "income"),
        ("What were the total operating expenses for {company} in {quarter} {year}?", "Total operating expenses", "expenses"),
        ("How much cash and cash equivalents did {company} hold in {quarter} {year}?", "Cash and cash equivalents", "cash"),
        ("What was the gross margin for {company} during {year} {quarter}?", "Gross margin", "margin"),
        ("Report the diluted earnings per share for {company} in {year} {quarter}.", "Diluted earnings per share", "eps"),
        ("What was {company}'s research and development expense in {year} {quarter}?", "Research and development", "r&d")
    ]
    
    # We want pages that are likely to contain dense financial tables
    # Filtering for pages with multiple tables or dense text blocks
    eligible_pages = df[(df['num_tables'] > 0) | (df['num_text_blocks'] > 20)]
    
    queries = []
    
    for i in range(num_queries):
        if len(eligible_pages) == 0:
            break
            
        row = eligible_pages.sample(1).iloc[0]
        template, keyword, category = random.choice(templates)
        
        company = row['ticker']
        year = row['fiscal_year']
        quarter = row['quarter']
        page_id = row['page_id']
        
        query_text = template.format(company=company, year=year, quarter=quarter)
        
        queries.append({
            "page_id": page_id,
            "query": query_text,
            "category": category,
            "difficulty": "synthetic_baseline"
        })
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test.json")
    
    with open(out_path, 'w') as f:
        json.dump(queries, f, indent=2)
        
    print(f"Successfully generated {len(queries)} synthetic queries at {out_path}")

if __name__ == "__main__":
    generate_synthetic_visfinqa(
        metadata_path="data/vision_rag/page_metadata.parquet",
        output_dir="data/vision_rag/visfinqa",
        num_queries=500
    )
