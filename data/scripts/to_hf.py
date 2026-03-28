from datasets import load_dataset
"""
Push ilocal JSONL filet to Hugging Face
Replace 'your-username' with your actual HF username
"""

def main():
    datasets_to_upload = [
        ("data/merged/LoL19.jsonl", "your-username/LoL19"),
        ("data/merged/LoL1921.jsonl", "your-username/LoL19-21")
    ]

    for local_file, repo_id in datasets_to_upload:
        print(f"\n--- Processing {repo_id} ---")
        print(f"Loading local dataset from {local_file}...")
        
        dataset = load_dataset(
            "json", 
            data_files=local_file, 
            split="train"
        )

        print(f"Pushing to Hugging Face Hub at {repo_id}...")
        dataset.push_to_hub(
            repo_id, 
            private=True  # private
        )
        
    print("\nAll uploads complete!")

if __name__ == "__main__":
    main()