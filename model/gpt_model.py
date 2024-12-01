import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .config_handler import load_config


def remove_last_unfinished_sentence(text: str) -> str:
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    
    # Check if the last sentence is unfinished
    if sentences and not re.match(r'.*[.!?]$', sentences[-1]):
        sentences.pop()
    return ' '.join(sentences)


# Generate output with rugpt3small_based_on_gpt2
# Input: params
# Output: list of responses
def generate_sync(text: str, do_sample=True, max_length=50, 
                  repetition_penalty=5.0,
                  top_k=5, top_p=0.95, temperature=1,
                  num_beams=None,
                  no_repeat_ngram_size=3) -> str:
    model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
    input_ids = tokenizer.encode(text, return_tensors="pt").cuda()

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
    
    return remove_last_unfinished_sentence(list(map(tokenizer.decode, out))[0])[len(text):].strip()


async def generate_gpt(text: str, config='config_gpt.json',
                   do_sample=True, max_length=50, repetition_penalty=5.0,
                   top_k=5, top_p=0.95, temperature=1,
                   num_beams=None,
                   no_repeat_ngram_size=3) -> str:
    data = load_config(config)
    if data is not None:
        do_sample = data.get('do_sample', do_sample)
        max_length = data.get('max_length', max_length)
        repetition_penalty = data.get('repetition_penalty', repetition_penalty)
        top_k = data.get('top_k', top_k)
        top_p = data.get('top_p', top_p)
        temperature = data.get('temperature', temperature)
        num_beams = data.get('num_beams', num_beams)
        no_repeat_ngram_size = data.get('no_repeat_ngram_size', no_repeat_ngram_size)
        
    executor = ThreadPoolExecutor()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, generate_sync, 
                                      text, do_sample, max_length, repetition_penalty,
                                      top_k, top_p, temperature, num_beams, no_repeat_ngram_size)