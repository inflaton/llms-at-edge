import os
import sys
import time
import torch
from dotenv import find_dotenv, load_dotenv

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)

from llm_toolkit.llm_utils import *
from llm_toolkit.data_utils import *

device = check_gpu()
is_cuda = torch.cuda.is_available()


def evaluate_model_with_num_shots(
    model_name,
    data_path,
    results_path,
    gguf_file_path=None,
    adapter_name_or_path=None,
    start_num_shots=0,
    end_num_shots=10,
    range_num_shots=[0, 1, 3, 5, 10],
    batch_size=1,
    max_new_tokens=2048,
    device="cuda",
    input_column="Headline_Details",
    output_column="Summarized_label",
    result_column_name=None,
    max_entries=0,
    load_in_4bit=False,
    using_llama_factory=False,
    using_vllm=False,
    repetition_penalty=1.1,
    debug=False,
):
    print(f"Evaluating model: {model_name} on {device}")

    if is_cuda:
        torch.cuda.empty_cache()
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"(0) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        torch.cuda.empty_cache()

    model, tokenizer = load_model(
        model_name,
        load_in_4bit=load_in_4bit,
        using_llama_factory=using_llama_factory,
        using_vllm=using_vllm,
        adapter_name_or_path=adapter_name_or_path,
        gguf_file_path=gguf_file_path,
    )

    if is_cuda:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"(2) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    if adapter_name_or_path is not None:
        model_name += "/" + adapter_name_or_path.split("/")[-1]

    for num_shots in range_num_shots:
        if num_shots < start_num_shots:
            continue
        if num_shots > end_num_shots:
            break

        print(f"*** Evaluating with num_shots: {num_shots}")

        datasets = prepare_dataset(
            data_path,
            input_column,
            output_column,
            tokenizer=tokenizer,
            num_shots=num_shots,
            get_prompt_templates=get_prompt_templates,
            max_entries=max_entries,
            for_openai=using_vllm,
        )
        if debug:
            print_row_details(datasets["test"].to_pandas())

        start_time = time.time()  # Start time

        predictions = eval_model(
            model,
            tokenizer,
            datasets["test"],
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            debug=debug,
        )

        end_time = time.time()  # End time
        exec_time = end_time - start_time  # Execution time
        print(f"*** Execution time for num_shots {num_shots}: {exec_time:.2f} seconds")

        model_name_with_shots = (
            result_column_name
            if result_column_name
            else f"{model_name}/shots-{num_shots:02d}({exec_time / len(datasets['test']):.3f})"
        )

        try:
            on_num_shots_step_completed(
                model_name_with_shots,
                datasets["test"],
                output_column,
                predictions,
                results_path,
            )
        except Exception as e:
            print(e)

    if is_cuda:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"(3) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    gguf_file_path = os.getenv("GGUF_FILE_PATH")
    adapter_name_or_path = os.getenv("ADAPTER_NAME_OR_PATH")
    load_in_4bit = os.getenv("LOAD_IN_4BIT") == "true"
    data_path = os.getenv("DATA_PATH")
    results_path = os.getenv("RESULTS_PATH")
    batch_size = int(os.getenv("BATCH_SIZE", 1))
    using_llama_factory = os.getenv("USING_LLAMA_FACTORY") == "true"
    using_vllm = os.getenv("USING_VLLM") == "true"
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 2048))
    start_num_shots = int(os.getenv("START_NUM_SHOTS", 0))
    end_num_shots = int(os.getenv("END_NUM_SHOTS", 10))

    print(
        model_name,
        adapter_name_or_path,
        load_in_4bit,
        data_path,
        results_path,
        using_llama_factory,
        max_new_tokens,
        batch_size,
    )

    evaluate_model_with_num_shots(
        model_name,
        data_path,
        results_path,
        gguf_file_path=gguf_file_path,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_entries=0,
        device=device,
        start_num_shots=start_num_shots,
        end_num_shots=end_num_shots,
        range_num_shots=[0, 1, 2, 4, 8, 10],
        load_in_4bit=load_in_4bit,
        adapter_name_or_path=adapter_name_or_path,
        using_llama_factory=using_llama_factory,
        using_vllm=using_vllm,
    )
