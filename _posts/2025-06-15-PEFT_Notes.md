
---
title: 'Parameter Efficient Fine Tuning Notes'
description: Explaination of different Paramete Efficient Finetuning Techniques
date: 2025-06-15 00:55:00 +0530
author: skrohit
categories: [peft, finetuning]
tags: [llms, finetuning, transformers]
pin: true
---

# Parameter Efficient Fine Tuning (PEFT)
## Making Large Language Models Adaptable Without Breaking the Bank 💰

---

## 🎯 Learning Objectives
By the end of this presentation, you will:
- Understand why PEFT is crucial for modern AI applications
- Master 4 key PEFT techniques: Prompt Tuning, Prefix Tuning, LoRA, and QLoRA
- Implement these techniques with hands-on code examples
- Understand quantization fundamentals for QLoRA

---

## 🤔 Opening Question
**Think about this:** If a pre-trained GPT model has 175 billion parameters, and you want to fine-tune it for your specific task, what challenges might you face?

*Take 30 seconds to think about it*

---

## 📊 The Problem: Traditional Fine-Tuning

### 🔢 Quick Math Challenge - Model Memory Requirements
If each parameter is 16 bits (2 bytes), how much storage would you need for:
- Model weights: 175B × 2 bytes = ?
- Gradients: 175B × 2 bytes = ?
- Optimizer states: 175B × 8 bytes = ?

**Total storage needed = ?**

*Why have we multipled by 8 for optimizer states*

### 🧮 Quick Math Challenge - Model Computational Requirements
If one forward-backward pass with a single token requires ~6 operations per parameter, calculate:
- Operations per token: 175B × 6 = ?
- For a batch of 32 sequences with 512 tokens each: ? operations
- If your GPU computes at 100 TFLOPS (10¹⁴ operations/s), how many seconds per batch?
- How many kilowatt-hours to train on 1 trillion tokens? (Assuming 0.1 kWh per 10¹⁸ operations)

*Why do we need ~6 operations per parameter in a forward-backward pass?*

### Traditional Approach Issues:
- **Memory Explosion**: Full fine-tuning requires storing gradients for ALL parameters
- **Storage Nightmare**: Need to save entire model for each task
- **Computational Cost**: Training 175B parameters = 💸💸💸
- **Catastrophic Forgetting**: Model forgets previously learned knowledge

---

## 💡 Enter PEFT: The Solution

**Parameter Efficient Fine Tuning** adapts large models by:
- Keeping original weights **frozen** ❄️
- Adding small, trainable modules
- Achieving similar performance with <1% trainable parameters

### 🎪 PEFT vs Full Fine-Tuning

| Aspect | Full Fine-Tuning | PEFT |
|--------|------------------|------|
| Trainable Parameters | 175B (100%) | 1.75M (0.01%) |
| Memory Usage | ~1.4TB | ~14GB |
| Training Time | Days/Weeks | Hours |
| Storage per Task | 350GB | 3.5MB |

---

## 🎯 PEFT Technique #1: Soft Prompt Tuning

### 🤔 Curious Question
*"What if we could teach a model new tasks just by changing how we ask questions?"*

### Core Concept
Instead of changing model weights, we learn **soft prompts** - trainable embeddings that guide the model's behavior.

```
Traditional: "Translate to French: Hello world"
Soft Prompt: [LEARNABLE_TOKENS] + "Hello world"
```

### 🔧 Implementation Challenge
Complete this PyTorch implementation:

```python
import torch
import torch.nn as nn

class SoftPromptTuning(nn.Module):
    def __init__(self, model, prompt_length, embedding_dim):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length
        
        # TODO: Initialize learnable prompt embeddings
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, embedding_dim) * 0.1
        )
        
        # Freeze the original model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        
        # TODO: Get input embeddings from the model
        input_embeds = self.model.get_input_embeddings()(input_ids)
        
        # TODO: Expand soft prompt for batch
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # TODO: Concatenate prompt with input
        full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # TODO: Create attention mask for full sequence
        prompt_attention = torch.ones(
            batch_size, self.prompt_length, device=attention_mask.device
        )
        full_attention = torch.cat([prompt_attention, attention_mask], dim=1)
        
        return self.model(inputs_embeds=full_embeds, 
                         attention_mask=full_attention)

# Usage example
model = # Your pre-trained model
soft_prompt_model = SoftPromptTuning(model, prompt_length=20, embedding_dim=768)
```

### 💭 Discussion Points
1. What are the advantages of soft prompts over hard prompts?
2. How might prompt length affect performance?
3. Can you think of tasks where this approach might struggle?

---

## 🧠 Quiz: Soft Prompt Tuning

### Question 1
What percentage of parameters are typically trainable in soft prompt tuning?
- a) 50%
- b) 10%
- c) 1%
- d) 0.01%

### Question 2
Which component is learned during soft prompt tuning?
- a) Model weights
- b) Attention mechanisms
- c) Prompt embeddings
- d) Loss function

### Question 3 (Code Challenge)
What's missing in this code?
```python
soft_prompt = nn.Parameter(torch.randn(10, 768))
# Missing: ___________
optimizer = torch.optim.Adam([soft_prompt], lr=0.001)
```

**Answers:** 1-d, 2-c, 3-Freezing original model parameters

---

## 🎯 PEFT Technique #2: Prefix Tuning

### 🤔 Curious Question
*"What if we could control the model's 'thinking process' by adding trainable prefixes to its internal representations?"*

### Core Concept
- Add trainable prefix vectors to key-value pairs in attention layers
- Model learns task-specific "context" at each layer
- More expressive than prompt tuning

### 📊 Prefix Tuning Architecture

```
Layer 1: [PREFIX_K, PREFIX_V] + [ORIGINAL_K, ORIGINAL_V]
Layer 2: [PREFIX_K, PREFIX_V] + [ORIGINAL_K, ORIGINAL_V]
...
Layer N: [PREFIX_K, PREFIX_V] + [ORIGINAL_K, ORIGINAL_V]
```

### 🔧 Implementation Challenge
```python
class PrefixTuning(nn.Module):
    def __init__(self, model, prefix_length, num_layers, hidden_size):
        super().__init__()
        self.model = model
        self.prefix_length = prefix_length
        
        # TODO: Create prefix parameters for each layer
        # Hint: Need separate prefixes for keys and values
        self.prefix_keys = nn.ParameterList([
            nn.Parameter(torch.randn(prefix_length, hidden_size) * 0.1)
            for _ in range(num_layers)
        ])
        
        self.prefix_values = nn.ParameterList([
            # TODO: Complete this
        ])
        
        # Freeze original model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        # TODO: Implement forward pass with prefix injection
        # This requires modifying attention computation
        pass

# Simplified usage with Hugging Face
from transformers import GPT2LMHeadModel
from peft import PrefixTuningConfig, get_peft_model

model = GPT2LMHeadModel.from_pretrained("gpt2")
config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=30,  # prefix length
    token_dim=768,
    num_transformer_submodules=2
)
model = get_peft_model(model, config)
```

### 🎮 Interactive Exercise
**Scenario**: You're building a chatbot for customer service. How would prefix tuning help compared to prompt tuning?

**

---

## 🧠 Quiz: Prefix Tuning

### Question 1
Where are prefix parameters added in prefix tuning?
- a) Input embeddings only
- b) Output layer only
- c) Key-Value pairs in attention layers
- d) Loss computation

### Question 2 (True/False)
Prefix tuning typically requires more parameters than soft prompt tuning.

### Question 3 (Code Debug)
Find the bug:
```python
prefix_keys = nn.Parameter(torch.randn(10, 768))
prefix_values = nn.Parameter(torch.randn(10, 768))
# Bug: These should be ParameterList for multiple layers!
```

**Answers:** 1-c, 2-True, 3-Should use ParameterList for multiple layers

---

## 🎯 PEFT Technique #3: Low Rank Adaptation (LoRA)

### 🤔 Curious Question
*"What if most of the knowledge needed for a new task already exists in the model, and we just need to make small adjustments?"*

### Core Concept: Matrix Decomposition Magic

Instead of updating weight matrix W directly:
```
W_new = W_original + ΔW
```

LoRA approximates ΔW as a low-rank decomposition:
```
ΔW = A × B  (where A is m×r and B is r×n, r << min(m,n))
```

### 📊 Visual Representation

```
Original Layer: X → W → Y
LoRA Layer:     X → W → Y
                ↓     ↑
                A → B (low-rank path)
```

### 🔧 Implementation Deep Dive

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # Get dimensions
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # TODO: Initialize LoRA matrices A and B
        # Hint: A should be (out_features, rank), B should be (rank, in_features)
        self.lora_A = nn.Parameter(torch.randn(out_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        
        # TODO: Initialize A with Kaiming normal, B with zeros
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.alpha / self.rank
    
    def forward(self, x):
        # TODO: Implement forward pass
        # original_output = ?
        # lora_output = ?
        # return original_output + lora_output * scaling
        
        original_output = self.original_layer(x)
        lora_output = self.dropout(x) @ self.lora_B.T @ self.lora_A.T
        return original_output + lora_output * self.scaling

# Practical LoRA with Hugging Face
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,          # scaling parameter
    target_modules=["c_attn", "c_proj"],  # which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
```

### 🎯 Parameter Efficiency Calculator

**Challenge**: Calculate trainable parameters for different ranks

```python
def calculate_lora_params(original_dim, rank, num_layers):
    """
    TODO: Complete this function
    Each LoRA layer adds: rank * (in_features + out_features) parameters
    """
    params_per_layer = rank * (original_dim + original_dim)
    total_params = params_per_layer * num_layers
    return total_params

# Test your function
original_params = 175_000_000_000  # 175B
lora_params = calculate_lora_params(4096, rank=16, num_layers=96)
efficiency = (lora_params / original_params) * 100

print(f"Parameter efficiency: {efficiency:.4f}%")
```

### 💡 Advanced LoRA Concepts

1. **Rank Selection**: Higher rank = more capacity but more parameters
2. **Target Modules**: Usually attention layers (Q, K, V, O)
3. **Scaling Factor**: α/r ratio affects adaptation strength

---

## 🧠 Quiz: LoRA Deep Dive

### Question 1
If a linear layer has dimensions 1024×1024 and LoRA rank=8, how many parameters does LoRA add?
- a) 8,192
- b) 16,384
- c) 32,768
- d) 1,048,576

### Question 2 (Code Challenge)
Complete the LoRA forward pass:
```python
def forward(self, x):
    original = self.original_layer(x)
    # TODO: Complete LoRA computation
    lora = x @ self.lora_B.T @ _____ * self.scaling
    return original + lora
```

### Question 3 (Conceptual)
Why is matrix B initialized to zeros in LoRA?

**Answers:** 1-b (8×2048), 2-self.lora_A.T, 3-To ensure initial LoRA output is zero

---

## 📊 Quantization Fundamentals
### Setting the Stage for QLoRA

### 🤔 Curious Question
*"How can we represent a 32-bit number using only 4 bits without losing too much information?"*

### What is Quantization?
Converting high-precision numbers (FP32) to lower precision (INT8, INT4) to save memory and computation.

### Types of Quantization

1. **Post-Training Quantization (PTQ)**
   ```python
   # Convert trained FP32 model to INT8
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

2. **Quantization-Aware Training (QAT)**
   ```python
   # Train with quantization in mind
   model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
   prepared_model = torch.quantization.prepare_qat(model)
   ```

### 🔢 Quantization Math Challenge

**Linear Quantization Formula:**
```
quantized_value = round((float_value - zero_point) / scale)
```

**Your Task**: Implement quantization and dequantization

```python
def quantize_tensor(tensor, bits=8):
    """
    TODO: Implement linear quantization
    Steps:
    1. Find min and max values
    2. Calculate scale and zero_point
    3. Quantize values
    """
    qmin = 0  # for unsigned quantization
    qmax = (2 ** bits) - 1
    
    min_val = tensor.min()
    max_val = tensor.max()
    
    # TODO: Calculate scale
    scale = (max_val - min_val) / (qmax - qmin)
    
    # TODO: Calculate zero_point
    zero_point = qmin - min_val / scale
    zero_point = torch.clamp(zero_point.round(), qmin, qmax)
    
    # TODO: Quantize
    quantized = torch.clamp(
        torch.round(tensor / scale + zero_point), qmin, qmax
    )
    
    return quantized, scale, zero_point

def dequantize_tensor(quantized, scale, zero_point):
    """TODO: Implement dequantization"""
    return scale * (quantized - zero_point)
```

### 📊 Memory Savings Visualization

| Precision | Bits | Memory for 1B params | Relative Size |
|-----------|------|---------------------|---------------|
| FP32      | 32   | 4 GB                | 100%          |
| FP16      | 16   | 2 GB                | 50%           |
| INT8      | 8    | 1 GB                | 25%           |
| INT4      | 4    | 500 MB              | 12.5%         |

---

## 🎯 PEFT Technique #4: QLoRA (Quantized LoRA)

### 🤔 Curious Question
*"What if we could get the benefits of LoRA while using 4x less memory for the base model?"*

### QLoRA Innovation
Combines the best of both worlds:
- **Base model**: Quantized to 4-bit (frozen)
- **LoRA adapters**: Full precision (trainable)
- **Gradients**: Computed through dequantization

### 🔧 QLoRA Architecture

```
Input → Quantized Base Model → Dequantize → LoRA Adapters → Output
         (4-bit, frozen)                    (FP16, trainable)
```

### Implementation with BitsAndBytes

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# TODO: Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # Nested quantization
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-large",
    quantization_config=bnb_config,
    device_map="auto"
)

# TODO: Add LoRA configuration
lora_config = LoraConfig(
    r=64,                    # Higher rank for complex tasks
    lora_alpha=128,         # 2x rank for scaling
    target_modules=[
        "c_attn",           # Attention weights
        "c_proj",           # Projection weights
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Combine quantization + LoRA
model = get_peft_model(model, lora_config)

# Check parameter efficiency
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
```

### 🎮 Hands-On Challenge: QLoRA Fine-Tuning

Complete this training loop:

```python
from transformers import TrainingArguments, Trainer

# TODO: Configure training arguments
training_args = TrainingArguments(
    output_dir="./qlora-results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    optim="paged_adamw_32bit",  # Memory-efficient optimizer
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    # TODO: Add more arguments
)

# TODO: Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # TODO: Add data collator
)

# Train the model
trainer.train()

# Save only LoRA weights (a few MB!)
model.save_pretrained("./my-qlora-adapter")
```

### 💡 QLoRA Advanced Features

1. **Normal Float 4-bit (NF4)**: Optimized for normal distributions
2. **Double Quantization**: Quantize quantization constants
3. **Paged Optimizers**: Handle memory spikes during training

---

## 🧠 Final Quiz: QLoRA Mastery

### Question 1
What's the typical memory reduction when using 4-bit quantization?
- a) 2x
- b) 4x
- c) 8x
- d) 16x

### Question 2 (Code Challenge)
Which quantization type is best for neural network weights?
```python
bnb_config = BitsAndBytesConfig(
    bnb_4bit_quant_type="___",  # fp4 or nf4?
)
```

### Question 3 (Scenario)
You have a 70B parameter model and want to fine-tune it on a single 24GB GPU. Which approach would work best?
- a) Full fine-tuning
- b) LoRA with FP16
- c) QLoRA with 4-bit
- d) Prompt tuning only

**Answers:** 1-b, 2-nf4, 3-c

---

## 🎯 PEFT Techniques Comparison

### Interactive Comparison Table

| Technique | Trainable Params | Memory Usage | Performance | Use Case |
|-----------|------------------|--------------|-------------|----------|
| **Prompt Tuning** | ~0.01% | Lowest | Good for simple tasks | Few-shot adaptation |
| **Prefix Tuning** | ~0.1% | Low | Better context control | Dialogue systems |
| **LoRA** | ~0.1-1% | Medium | High performance | General fine-tuning |
| **QLoRA** | ~0.1-1% | Lowest | High performance | Large models, limited GPU |

### 🎮 Decision Tree Exercise

**Scenario-Based Challenges:**

1. **Startup with limited budget**: You have a 13B model and need to adapt it for 5 different tasks. GPU memory: 16GB.
   - *What would you choose and why?*

2. **Research lab**: You want to understand how different layers contribute to task performance.
   - *Which technique gives you the most insights?*

3. **Production system**: You need to serve 100 different fine-tuned versions of the same model.
   - *How would you optimize for storage and serving?*

---

## 🛠️ Practical Implementation

### 🎯 Mini-Project: Build Your Own PEFT Pipeline

```python
class PEFTComparison:
    def __init__(self, base_model_name):
        self.base_model_name = base_model_name
        self.models = {}
    
    def add_prompt_tuning(self, prompt_length=20):
        """TODO: Implement prompt tuning setup"""
        pass
    
    def add_lora(self, rank=16, alpha=32):
        """TODO: Implement LoRA setup"""
        pass
    
    def add_qlora(self, rank=64, alpha=128):
        """TODO: Implement QLoRA setup"""
        pass
    
    def compare_efficiency(self):
        """Compare parameter efficiency across techniques"""
        results = {}
        for name, model in self.models.items():
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() 
                          if p.requires_grad)
            results[name] = {
                'total': total,
                'trainable': trainable,
                'efficiency': trainable / total * 100
            }
        return results

# Your task: Complete the implementation!
```

### 🎪 Live Coding Challenge

Work in pairs to implement one PEFT technique from scratch. You have 15 minutes!

---

## 🚀 Advanced Topics & Future Directions

### 🤔 Curious Questions for Exploration

1. **AdaLoRA**: What if we could dynamically adjust LoRA ranks during training?
2. **LoRA+**: Can we improve LoRA by using different learning rates for A and B matrices?
3. **MultiModal PEFT**: How do we adapt vision-language models efficiently?

### 🔬 Research Frontiers

```python
# Emerging techniques (pseudocode)
class AdaptiveLoRA(nn.Module):
    """Dynamically prune LoRA ranks based on importance"""
    def __init__(self, base_layer, max_rank=64):
        # TODO: Implement adaptive ranking
        pass

class LoRAPlus(nn.Module):
    """Different learning rates for LoRA matrices"""
    def __init__(self, base_layer, rank=16, lr_ratio=16):
        # TODO: Implement LoRA+ training
        pass
```

---

## 🎯 Real-World Case Studies

### Case Study 1: Hugging Face's PEFT Library
- **Challenge**: Make PEFT accessible to everyone
- **Solution**: Unified API for all PEFT methods
- **Impact**: 10,000+ models fine-tuned with PEFT

### Case Study 2: Alpaca Model
- **Challenge**: Create instruction-following model cheaply
- **Solution**: QLoRA fine-tuning of LLaMA
- **Result**: GPT-3.5 level performance for $600

### 🎮 Your Turn: Design a PEFT Strategy

**Scenario**: You're tasked with creating a multilingual customer service chatbot for an e-commerce company.

**Constraints**:
- Base model: 7B parameters
- GPU budget: Single A100 (40GB)
- Languages: English, Spanish, French, German, Chinese
- Response time: <200ms

**Your Task**: Design a PEFT strategy that addresses:
1. Which PEFT technique(s) to use?
2. How to handle multiple languages?
3. How to ensure fast inference?
4. How to update for new languages?

*Present your solution in groups!*

---

## 📊 Performance Benchmarks

### 🎯 Interactive Benchmark Analysis

| Task | Full Fine-Tuning | LoRA | QLoRA | Prompt Tuning |
|------|------------------|------|-------|---------------|
| **GLUE** | 85.2 | 84.8 | 84.5 | 82.1 |
| **SuperGLUE** | 71.5 | 70.9 | 70.2 | 67.8 |
| **Code Generation** | 89.3 | 88.7 | 88.1 | 85.2 |
| **Dialogue** | 92.1 | 91.8 | 91.3 | 89.7 |

**Discussion Questions**:
1. Why does QLoRA perform slightly worse than LoRA?
2. When might prompt tuning be preferred despite lower scores?
3. How would you choose between techniques for your specific use case?

---

## 🎓 Wrap-Up Challenge: PEFT Jeopardy!

### 🎪 Final Interactive Game

**Category: PEFT Fundamentals**
- *Answer*: "This technique adds learnable embeddings to the input sequence"
- *Question*: What is Prompt Tuning?

**Category: Implementation**
- *Answer*: "The parameter that controls the magnitude of LoRA adaptations"
- *Question*: What is alpha (scaling factor)?

**Category: Efficiency**
- *Answer*: "The typical percentage of parameters that are trainable in LoRA"
- *Question*: What is 0.1-1%?

## 🎯 Key Takeaways

### 🧠 Remember These Principles:

1. **PEFT isn't one-size-fits-all** - Choose based on your constraints
2. **Parameter efficiency ≠ Performance loss** - Often matches full fine-tuning
3. **Memory is the main bottleneck** - QLoRA makes 70B+ models accessible
4. **Composability is powerful** - Combine multiple LoRA adapters
5. **The field is rapidly evolving** - Stay updated with latest techniques

### 🤔 Final Reflection Question

*"Given what you've learned today, how would you approach fine-tuning a 175B parameter model for your dream AI application with a $1000 budget?"*

*Take 5 minutes to write your strategy...*

---

## 📚 Resources for Continued Learning

### Essential Papers
- **LoRA**: "Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA**: "Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **Prefix Tuning**: "Optimizing Continuous Prompts" (Li & Liang, 2021)

### Code Repositories
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [LoRA Implementation](https://github.com/microsoft/LoRA)

### Interactive Tutorials
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Colab Notebook](https://colab.research.google.com/...)
- [PEFT Examples](https://github.com/huggingface/peft/tree/main/examples)

---

## 🎉 Thank You!


### 🔮 What's Next?
- **Next Session**: "Retrieval Augmented Generation (RAG)"

**Remember**: The best way to learn PEFT is to practice PEFT! 🚀

---

*"Parameter efficiency is not about doing less, it's about doing more with less."*