import copy
import torch
import argparse
import transformers



def generate_text(query_text, messages, pipeline):
    content = template_message[-1]["content"] + "\n" + query_text
    messages[-1]["content"] = content
    output = pipeline(messages, num_return_sequences=1)
    simplified_text = output[0]['generated_text'][-1]["content"]
    return simplified_text
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    args = parser.parse_args()


    model_name = "meta-llama/Meta-Llama-3-8B-instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # token=HF_TOKEN,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, 
        # token=HF_TOKEN, 
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )

    template_message = [
        {
            "role": "system",
            "content": "You are a professional editor trying to simplify difficult Finnish news articles for non-native speakers at intermediate level of learning Finnish.", 
        },
        {
            "role": "user",
            "content": "Please simplify the following news article in Finnish. The output text should be in Finnish.",
        },
    ]
    

    file_context = open(args.input_file).read().strip()
    print(generate_text(file_context, copy.deepcopy(template_message), pipeline))
    