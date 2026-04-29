from huggingface_hub import hf_hub_list

files = hf_hub_list(
    "nvidia/PhysicalAI-SmartSpaces",
    repo_type="dataset",
    path="MTMC_Tracking_2025/train/Warehouse_000",
)
print(len(files))
for f in files[:100]:
    print(f)
