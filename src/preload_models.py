from transformers import (
    AutoImageProcessor, 
    AutoModelForObjectDetection,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

print("Téléchargement des modèles...")

# DETR pour détection d'objets
print("1. DETR...")
AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Qwen pour description de scènes
print("2. Qwen VLM...")
AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

print("✅ Tous les modèles sont téléchargés !")

