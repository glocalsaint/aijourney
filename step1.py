# Install required libs (if not already installed)
# pip install torch transformers accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Model repo for quantized Mistral 7B (you can swap for another quantized checkpoint)
# model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"  
model_name = "microsoft/phi-2"  # Example model, replace with a quantized version if available

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in 4-bit (quantized)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",       
    trust_remote_code=True,
)

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="cpu"
)

# Try it out
prompt = "Explain quantum computing in simple terms:"
output = generator(prompt, max_new_tokens=200, temperature=0.7)

print(output[0]["generated_text"])
