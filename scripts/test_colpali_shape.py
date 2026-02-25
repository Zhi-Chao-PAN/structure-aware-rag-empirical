import torch
from src.vision_rag.model.backbone import ColPaliBackbone
from PIL import Image

def main():
    print("Loading ColPali...")
    # Create an empty backbone (bfloat16) to check memory
    backbone = ColPaliBackbone()
    backbone.eval()
    
    print("Loaded. Testing forward pass...")
    # Create a dummy image
    img = Image.new("RGB", (1024, 1024), color="white")
    
    with torch.no_grad():
        # Test encode_image
        print("Encoding image...")
        E_d = backbone.encode_image([img])
        print(f"Image embeddings shape: {E_d.shape}")
        
        print("Encoding query...")
        query = "What is the revenue?"
        E_q = backbone.encode_query([query])
        print(f"Query embeddings shape: {E_q.shape}")
        
        # Check tokenization of query
        inputs = backbone.processor.process_queries([query])
        input_ids = inputs["input_ids"][0]
        tokens = backbone.processor.tokenizer.convert_ids_to_tokens(input_ids)
        print(f"Query tokens: {tokens}")

if __name__ == "__main__":
    main()
