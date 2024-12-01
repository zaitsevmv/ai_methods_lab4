import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .config_handler import load_config


# Remove the last unfinished sentence from the text
def remove_last_unfinished_sentence(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text.strip())  # Split text into sentences
    if sentences and not re.match(r'.*[.!?]$', sentences[-1]):  # Check if last sentence is unfinished
        sentences.pop()  # Remove unfinished sentence
    return ' '.join(sentences)  # Join sentences back into a single string


# Generate output using the GPT-2 based model
def generate_sync(text: str, do_sample=True, max_length=50, 
                  repetition_penalty=5.0, top_k=5, top_p=0.95, 
                  temperature=1, num_beams=None, no_repeat_ngram_size=3) -> str:
    model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)  # Load tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()  # Load model and move to GPU
    input_ids = tokenizer.encode(text, return_tensors="pt").cuda()  # Encode input text

    # Generate text using the model
    out = model.generate(input_ids.cuda(),
                         max_length=max_length,
                         repetition_penalty=repetition_penalty,
                         do_sample=do_sample,
                         top_k=top_k,
                         top_p=top_p,
                         temperature=temperature,
                         num_beams=num_beams,
                         no_repeat_ngram_size=no_repeat_ngram_size,
                         num_return_sequences=1)
    
    # Decode and clean output
    output = remove_last_unfinished_sentence(list(map(tokenizer.decode, out))[0])[len(text):].strip() 
    if output == ' ':
        output = 'Возникла ошибка, попробуйте позже'
    return output


# Asynchronous function to generate text with GPT model
async def generate_gpt(text: str, config='config_gpt.json',
                   do_sample=True, max_length=50, repetition_penalty=5.0,
                   top_k=5, top_p=0.95, temperature=1,
                   num_beams=None, no_repeat_ngram_size=3) -> str:
    data = load_config(config)  # Load configuration from file
    if data is not None:
        # Update parameters from config if available
        do_sample = data.get('do_sample', do_sample)
        max_length = data.get('max_length', max_length)
        repetition_penalty = data.get('repetition_penalty', repetition_penalty)
        top_k = data.get('top_k', top_k)
        top_p = data.get('top_p', top_p)
        temperature = data.get('temperature', temperature)
        num_beams = data.get('num_beams', num_beams)
        no_repeat_ngram_size = data.get('no_repeat_ngram_size', no_repeat_ngram_size)
        
    executor = ThreadPoolExecutor()  # Create a thread pool executor
    loop = asyncio.get_event_loop()  # Get the current event loop
    return await loop.run_in_executor(executor, generate_sync,  # Run synchronous generation in executor
                                      text, do_sample, max_length, repetition_penalty,
                                      top_k, top_p, temperature, num_beams, no_repeat_ngram_size)