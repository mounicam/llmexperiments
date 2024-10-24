import copy
import glob
import torch
import argparse
import transformers


def generate_text(query_text, messages, pipeline):
    content = template_message[-1]["content"] + "\n" + query_text
    messages[-1]["content"] = content
    print(messages)
    print("**" * 40)
    output = pipeline(messages, num_return_sequences=1)
    simplified_text = output[0]['generated_text'][-1]["content"]
    print(simplified_text)
    print("**" * 40)
    return simplified_text
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file")
    parser.add_argument("--input_folder")
    parser.add_argument("--output_folder")
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
            "content": "You are a helpful AI assistant.", 
        },
        {
            "role": "user",
            "content": open(args.prompt_file).read().strip(),
        },
    ]
    
    
    for fpath in glob.glob(args.input_folder + "/*"):
        file_context = open(fpath).read().strip()
        simplified_text = generate_text(file_context, copy.deepcopy(template_message), pipeline)
        fname = fpath.split("/")[-1]
        with open(args.output_folder + fname, "w") as fp:
            fp.write(simplified_text)
    