from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

SCRUTINY_SAVE_DIR = "./scrutiny_data/"
PLOT_SAVE_DIR = "./plot_dense/"
os.environ["SCRUTINY_SAVE_DIR"] = SCRUTINY_SAVE_DIR


def main():
    model_name_or_path = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    text = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in "
        "May. How many clips did Natalia sell altogether in April and May?"
    ]

    model_input = tokenizer(text, return_tensors="pt")
    model_input = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in model_input.items()}
    with torch.no_grad():
        _ = model(**model_input)

    scrutiny_files = os.listdir(SCRUTINY_SAVE_DIR)
    for i, scrutiny_file in enumerate(scrutiny_files):
        scrutiny_file_path = os.path.join(SCRUTINY_SAVE_DIR, scrutiny_file)
        scrutiny_data = torch.load(scrutiny_file_path)
        active_proportion = scrutiny_data["active_proportion"].numpy()
        plt.plot(active_proportion)
        plt.savefig(os.path.join(PLOT_SAVE_DIR, f"dense_{i}.png"))
        plt.close()


if __name__ == "__main__":
    main()
