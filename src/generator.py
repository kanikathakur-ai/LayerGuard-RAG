import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import GENERATOR_MODEL, GENERATOR_MAX_NEW_TOKENS, GENERATOR_LOAD_IN_4BIT


_PROMPT_TEMPLATE = """\
Based on the following documents, answer the question with a short factual answer.

Documents:
{context}

Question: {query}
Answer:"""


def load_generator(model_name: str = GENERATOR_MODEL) -> tuple:
    """Load Llama-3.1-8B-Instruct with optional 4-bit quantization."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if GENERATOR_LOAD_IN_4BIT and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()
    return model, tokenizer


def generate_answer(
    query: str,
    context_docs: list[str],
    model,
    tokenizer,
    max_new_tokens: int = GENERATOR_MAX_NEW_TOKENS,
) -> str:
    """Generate a short factual answer given query and retrieved context documents."""
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(context_docs))
    prompt = _PROMPT_TEMPLATE.format(context=context, query=query)

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Trim to first sentence/line for short factual answers
    answer = answer.split("\n")[0].split(".")[0].strip()
    return answer
