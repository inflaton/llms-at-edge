import os
import sys
import time
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

def evaluate_model_with_num_shots(
    model_name,
    data_path,
    results_path,
    range_num_shots=[0, 1, 2, 4, 8, 10],
    start_num_shots=0,
    end_num_shots=10,
    max_entries=0,
    input_column="Headline_Details",
    output_column="gpt-4o_label",
    result_column_name=None,
    cerebras=False,
    ollama=False,
    debug=False,
):
    print(f"Evaluating model: {model_name}")

    datasets = prepare_dataset(
        data_path, input_column, output_column, max_entries=max_entries
    )
    # print_row_details(datasets["test"].to_pandas())

    for num_shots in range_num_shots:
        if num_shots < start_num_shots:
            continue
        if num_shots > end_num_shots:
            break

        print(f"* Evaluating with num_shots: {num_shots}")

        system_prompt, user_prompt = get_prompt_templates(
            datasets["train"], num_shots=num_shots, input_column=input_column
        )

        start_time = time.time()  # Start time

        predictions = eval_dataset_using_openai(
            system_prompt,
            user_prompt,
            datasets["test"],
            input_column,
            model=model_name,
            base_url=(
                "https://api.cerebras.ai/v1"
                if cerebras
                else "http://localhost:11434" if ollama else None
            ),
            api_key=(
                os.environ.get("CEREBRAS_API_KEY")
                if cerebras
                else "ollama" if ollama else os.environ.get("OPENAI_API_KEY")
            ),
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
                "gpt-4o_label",
                predictions,
                results_path,
            )
        except Exception as e:
            print(e)
            # print("Error saving results: ", predictions)
            # print(datasets["test"])


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    data_path = os.getenv("DATA_PATH")
    results_path = os.getenv("RESULTS_PATH")
    start_num_shots = int(os.getenv("START_NUM_SHOTS", 0))
    end_num_shots = int(os.getenv("END_NUM_SHOTS", 10))
    max_entries = int(os.getenv("MAX_ENTRIES", 0))
    ollama = os.getenv("OLLAMA") == "true"

    print(
        model_name,
        data_path,
        results_path,
        start_num_shots,
        end_num_shots,
        ollama,
    )
    
    check_gpu()

    evaluate_model_with_num_shots(
        model_name,
        data_path,
        results_path=results_path,
        start_num_shots=start_num_shots,
        end_num_shots=end_num_shots,
        ollama=ollama,
        max_entries=max_entries,
    )
