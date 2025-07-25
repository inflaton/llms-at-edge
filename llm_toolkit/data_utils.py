import os
import re
import json
import glob
import copy
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datasets import load_dataset, concatenate_datasets, Dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from llm_toolkit.llm_utils import load_tokenizer

print(f"loading {__file__}")

with open("dataset/categories.json", "r") as f:
    categories_json = json.load(f)
categories_list = list(categories_json.keys())


system_prompt_template = """Task: Classify Inputs into Predefined Categories

Your primary objective is to analyze the given input and assign it to one of the predefined categories: {categories_list}. Evaluate the content carefully and use the defining characteristics of each category to ensure an accurate classification.

Guidelines:
1. Understand the Categories:
Each category has specific attributes that distinguish it. Familiarize yourself with these attributes by referring to the category descriptions provided in the JSON below. Use these details to guide your classification:

{categories_json}

2. Contextual Analysis:
Consider the broader context of the input. If an input could potentially fit into multiple categories, select the one that most closely aligns with its primary intent or focus.
3. Handling Ambiguity:
For ambiguous inputs or those that do not clearly align with any category, choose the category that most closely matches the content provided.
4. Ensure Accuracy and Consistency:
Strive for consistent and accurate classifications. Avoid arbitrary or random assignments.
5. Provide Feedback:
If the input cannot be classified into any of the given categories, classify it as “Others.”

Instructions for Output:
1. Once the category is identified, provide “specific tags” by selecting from the list corresponding to the identified category, as defined in the JSON.
2. Ensure the selected “specific tags” accurately reflect the details and context of the input.

Output Format:

Return your classification in the following JSON format:

{{{{
  "category": "<Selected Category>",
  "specific_tags": ["<Selected Tag 1>", "<Selected Tag 2>", ...]
}}}}


"""

gpt_4o_generated_examples = """
- Input:

Local sources reported that operations at Pier 1 and 2 container terminals at the Port of Durban have suspended due to strong winds on December 27 from 18:50 (local time) and resumed at 23:10 on the same day. For Pier 2 terminal, operations stopped at 19:30 and resumed at 20:35 respectively.

- Output:

{{
  "category": "Weather",
  "specific_tags": ["Severe Winds"]
}}

- Input:

Information received states that emergency personnel are working to contain a blaze at Off Road Warehouse in commercial San Diego, on 17 November. It is detailed that the store is located at 7915 Balboa Avenue. Traffic maps show that Balboa Avenue is closed both ways between Mercury Street and Convoy Street. Travelers should use caution in the area and divert away from any encountered fire suppression operations.

- Output:

{{
  "category": "Administrative Issue",
  "specific_tags": ["Roadway Closure", "Public Safety Advisory"]
}}

- Input:

Protests against climate change are anticipated nationwide on 29 November and 6 December as part of the ‘Fridays for Future’ global climate strike. Specific details of planned events have not been confirmed, but are likely to occur in major cities across the country. Previous climate strikes have seen large turnout in cities such as New York City, Philadelphia, and Washington, D.C.

- Output:

{{
  "category": "Worker Strike",
  "specific_tags": ["Protest", "Civil Unrest Advisory"]
}}

- Input:

Government sources reported a fire at the Woolwich Dockyard, located near Belson Rd and Borgard Rd. No injuries were immediately reported. All rail lines from London towards Slade Green are running again. This incident is closed.

- Output:

{{
  "category": "Accident",
  "specific_tags": ["Non-industrial Fire"]
}}

- Input:

Local media sources indicated on November 30 that the Ekurhuleni Central Crime Intelligence Unit arrested 4 suspects and recovered computer printer equipment cargo from their November 21 truck theft at the corner of Main Reef Road and Ulysses Street in Cleveland. The truck was en route from Durban to Johannesburg when it was hijacked in Randfontein. The cargo was worth ZAR 5 million (EUR 309018.21; USD 352673.95), and some laptops are still missing. Distributors should be mindful of cargo theft risks in Randfontein and should plan accordingly.

- Output:

{{
  "category": "Terrorism",
  "specific_tags": ["Cargo Theft", "Organized Crime"]
}}

- Input:

Anonymous sources have reported that a ransomware attack has disrupted network operations for a major logistics provider. The attack occurred on November 15, and data breaches were confirmed, exposing sensitive customer and shipment details. The company has stated that recovery is underway but advised customers to expect delays.

- Output:

{{
  "category": "Cyber Attack",
  "specific_tags": ["Ransomware", "Data Breach"]
}}

- Input:

The Selangor Health Department reported that two students of a Secondary School in Pandamaran Jaya in Port Klang had been infected with COVID-19 virus.

- Output:

{{
  "category": "Others",
  "specific_tags": ["Outbreak of Disease"]
}}

- Input:

An incident of workplace negligence was reported at a construction site in downtown Chicago on November 19, where an unfastened scaffolding collapsed, injuring two workers. Investigations are ongoing to determine accountability.

- Output:

{{
  "category": "Human Error",
  "specific_tags": ["Workplace Accident"]
}}

- Input:

Shipping delays were reported at the Port of Los Angeles on December 1 due to a customs system outage. Containers requiring clearance were delayed for up to 12 hours, affecting supply chains across the region.

- Output:

{{
  "category": "Administrative Issue",
  "specific_tags": ["Customs Delay", "Port Disruption"]
}}

- Input:

Russian media sources are reporting that courts, schools, and hospitals across Saint Petersburg have been evacuated today due to anonymous threats. It is understood that people have been evacuated from Petrodvorets, Oktyabrsky, Kolpinsky, Petrogradsky, Kuibyshevsky, and Sestroretsky district courts. Furthermore, the State University of the Sea and River Fleet, St. Petersburg State University of Railway Engineering, Higher School of Folk Arts, St. Petersburg State University of Telecommunications, and S.M. Military Medical Academy Kirov have all been evacuated. This is the fourth consecutive week of evacuations from public buildings due to such threats. It is not known when the situation will normalize.

- Output:

{{
  "category": "Terrorism",
  "specific_tags": ["Bomb Threat", "Public Safety"]
}}

- Input:

A series of earthquakes shook the southern region of California on November 22. The most powerful tremor measured 6.4 on the Richter scale, causing landslides in rural areas and minor structural damage in towns nearby. Emergency services are on alert.

- Output:

{{
  "category": "Weather",
  "specific_tags": ["Earthquake", "Landslide"]
}}

"""


def find_nth_occurrence(string, substring, n):
    start = -1
    for _ in range(n):
        start = string.find(substring, start + 1)
        if start == -1:
            return -1
    return start


def get_prompt_templates(
    train_dataset=None,
    num_shots=0,
    input_column="Headline_Details",
    output_column="gpt-4o_label",
    remove_double_curly_brackets=False,
):
    print(
        f"Generating prompt templates for {num_shots} shots with {input_column} and {output_column}"
    )
    examples = "\nExample Inputs and Outputs:\n" if num_shots > 0 else ""

    if num_shots > 0:
        index = find_nth_occurrence(
            gpt_4o_generated_examples, "- Input:", num_shots + 1
        )
        examples += (
            gpt_4o_generated_examples[:index]
            if index >= 0
            else gpt_4o_generated_examples
        )
        # print("examples:", examples)

    # mappings = {}
    # j = 0
    # for i in range(num_shots):
    #     while j < len(train_dataset):
    #         if train_dataset[j]["Summarized_label"] not in mappings:
    #             mappings[train_dataset[j]["Summarized_label"]] = 1

    #             example_input = train_dataset[j][input_column]
    #             example_output = train_dataset[j][output_column]
    #             examples += f"\n\n- Input: {example_input}\n- Output: {example_output}"
    #             j += 1
    #             break
    #         else:
    #             j += 1

    #     if j >= len(train_dataset):
    #         break

    #     if (i + 1) % len(categories) == 0:
    #         mappings = {}

    # print("system_prompt_template:", system_prompt_template)
    system_prompt = system_prompt_template.format(
        categories_list=categories_list,
        categories_json="{" + f"{categories_json}" + "}",
    )
    system_prompt += examples

    if remove_double_curly_brackets:
        system_prompt = system_prompt.replace("{{", "{").replace("}}", "}")

    return (
        system_prompt,
        "- Input:\n\n{input}\n\n- Output:\n\n",
    )


f1_metric = evaluate.load("f1")

patten = re.compile(r"\"category\":\s*\"(.*?)\"")


def extract_answer(text, debug=False):
    if debug:
        print(f"extract_answer: {text}")
    if text:
        text = text.split("orrected output:")[-1]  # matching Corrected or corrected
        if "```" in text:
            text = text.split("```")[1]
            text = text.replace("json", "", 1).strip()
            text = text.replace("',", '",')
            text = text.replace("'\n", '"\n')
        text = text.strip()
        if debug:
            print("--------\nstep 0:", text)
        if text.startswith("{"):
            try:
                json_end = text.index("}")
                text = text[: json_end + 1]
                if debug:
                    print("--------\nstep 0a:", text)
                json_data = json.loads(text)
                if "category" in json_data:
                    return json_data["category"]
                print(f"category not in json: {text}")
            except:
                matches = patten.findall(text)
                if matches and len(matches) > 0:
                    text = matches[0]
                    if debug:
                        print("--------\nstep 0b:", text)
                else:
                    print(f"Error parsing json: {text}")

        text = text.replace("*", "").strip()
        text = text.split("Reasoning:")[0].strip() + "\n"

        pattern = re.compile(
            r"\b(best fit|closest fit|match|classify|classified).*?\b(is|as|the category:).+?\b(.*?)['|\"|\n]",
            re.DOTALL | re.MULTILINE,
        )
        if pattern.search(text):
            matches = pattern.search(text)
            text = matches.group(3)
            if debug:
                for i in range(1, len(matches.groups()) + 1):
                    print(f"--------\nstep 1a.{i}:", matches.group(i))
                print("--------\nstep 1a:", text)

        pattern = r"(falls|classif).+?\bunder\b.+?['|\"](.*?)['|\"|\n]"
        if re.search(pattern, text):
            text = re.search(pattern, text).group(2)
            if debug:
                print("--------\nstep 1b:", text)

        pattern = re.compile(
            r"\b(classif|category).*?\b[i|a]s.+?\b(.*?)['|\"|\n]",
            re.DOTALL | re.MULTILINE,
        )
        if pattern.search(text):
            text = pattern.search(text).group(2)
            if debug:
                print("--------\nstep 1c:", text)

        # Define the separators
        separators = r"\n|\.\s"
        text = re.split(separators, text)[0].strip()
        if debug:
            print("--------\nstep 2:", text)

        separators = r"[:(]"
        text = re.split(separators, text)[-1].strip()
        text = re.split(separators, text)[-1].strip()
        separators = r" or "
        text = re.split(separators, text)[0].strip()
        if debug:
            print("--------\nstep 3:", text)

        text = text.replace('"', "'").strip()
        text = text.replace(".", "").strip()

        if debug:
            print("--------\nstep 4:", text)

        parts = text.split("'")
        if len(parts) > 1:
            text = parts[-2].strip()
            if debug:
                print("--------\nstep 5:", text)

    return text


def check_invalid_categories(df, column, debug=False):
    count = 0
    for key in df.value_counts(column).keys():
        original_key = key
        text = key
        if "```" in text:
            text = text.split("```")[1]
            text = text.replace("json", "", 1).strip()
            text = text.replace("',", '",')
            text = text.replace("'\n", '"\n')
        key = text.strip()
        if key.startswith("{"):
            try:
                json_data = json.loads(key)
                if "category" in json_data:
                    key = json_data["category"]
            except:
                pass
        if key not in categories_list:
            count += df.value_counts(column)[original_key]

        if debug:
            cat = extract_answer(original_key)
            if cat not in categories_list:
                print(cat, "-->", original_key)

    if debug:
        print(column, " invalid categories: ", count)
        print("=" * 71)
    return count / len(df)


def calc_f1_score(references, predictions, debug=False):
    references = [
        categories_list.index(r) if r in categories_list else -1 for r in references
    ]
    predictions = [
        categories_list.index(p) if p in categories_list else -1 for p in predictions
    ]
    if debug:
        print("references:", references)
        print("predictions:", predictions)

    results = f1_metric.compute(
        predictions=predictions, references=references, average="weighted"
    )
    return results


def calc_metrics(references, predictions, post_process=True, debug=False):
    if debug and len(references) != len(predictions):
        print("references:", references)
        print("predictions:", predictions)
    elif debug:
        print("references[0]:", references[0])
        print("predictions[0]:", predictions[0])
    
    assert len(references) == len(
        predictions
    ), f"lengths are difference: {len(references)} != {len(predictions)}"

    if references[0].startswith("{"):
        references = [extract_answer(r, debug) for r in references]

    if post_process:
        predictions = [extract_answer(p, debug) for p in predictions]

    results = calc_f1_score(references, predictions, debug=debug)

    correct = [1 if ref == pred else 0 for ref, pred in zip(references, predictions)]
    accuracy = sum(correct) / len(references)

    results["accuracy"] = accuracy
    if debug:
        correct_ids = [i for i, c in enumerate(correct) if c == 1]
        results["correct_ids"] = correct_ids

    return results


def on_num_shots_step_completed(
    model_name, dataset, output_column, predictions, results_path
):
    save_results(
        model_name,
        results_path,
        dataset,
        predictions,
        debug=False,
    )

    metrics = calc_metrics(dataset[output_column], predictions, debug=False)
    print(f"{model_name} metrics: {metrics}")


tokenizers = {}


def get_tokenizer(model_name):
    if "qwen" in model_name.lower():
        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    else:
        model_name = "meta-llama/Llama-3.2-1B-Instruct"

    if model_name in tokenizers:
        return tokenizers[model_name]

    tokenizer = load_tokenizer(model_name)
    tokenizers[model_name] = tokenizer
    return tokenizer


def calc_num_tokens(model, num_shots, inputs, predictions, debug=False):
    tokenizer = get_tokenizer(model)
    tokens = 0
    system_prompt, user_prompt = get_prompt_templates(num_shots=num_shots)
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": None},
        {
            "role": "assistant",
            "content": None,
        },
    ]
    for input, pred in zip(inputs, predictions):
        prompt = user_prompt.format(input=input)
        messages[1] = {"role": "user", "content": prompt}
        messages[2]["content"] = pred
        encoded_prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        # print(f"encoded_prompt: {encoded_prompt}")
        tokens += len(encoded_prompt)
    return tokens


def get_metrics(
    df,
    result_col_start_idx=4,
    input_column="Headline_Details",
    label_column="Summarized_label",
    variant="shots",
    post_process=True,
    mean_eval_time=False,
    debug=False,
):
    df = df.copy()
    columns = df.columns[result_col_start_idx:]
    metrics_df = pd.DataFrame(columns.T)
    metrics_df.rename(columns={0: "model"}, inplace=True)
    metrics_df[variant] = metrics_df["model"].apply(
        lambda x: x.split(f"{variant}-")[-1].split("(")[0]
    )
    if variant != "rpp":
        metrics_df[variant] = metrics_df[variant].astype(int)

    if mean_eval_time:
        metrics_df["eval_time"] = metrics_df["model"].apply(
            lambda x: float(x.split("(")[-1].split(")")[0])
        )

    metrics_df["model"] = metrics_df["model"].apply(
        lambda x: x.split(f"/{variant}-")[0].split("/checkpoint")[0]
    )

    metrics_df.reset_index(inplace=True)
    metrics_df = metrics_df.drop(columns=["index"])

    accuracy_raw = []
    f1_raw = []
    accuracy = []
    f1 = []
    ratio_valid_categories = []
    total_tokens = []

    for col, model, shot in zip(columns, metrics_df["model"], metrics_df[variant]):
        try:
            metrics = calc_metrics(
                df[label_column], df[col], post_process=None, debug=debug
            )
        except:
            metrics = {"f1": 0, "accuracy": 0}

        print(f"{col} - metrics_raw: {metrics}")
        accuracy_raw.append(metrics["accuracy"])
        f1_raw.append(metrics["f1"])

        metrics = calc_metrics(
            df[label_column], df[col], post_process=post_process, debug=debug
        )

        print(f"{col} - metrics: {metrics}")
        accuracy.append(metrics["accuracy"])
        f1.append(metrics["f1"])

        invalid_categories = check_invalid_categories(df, col, debug=debug)
        ratio_valid_categories.append(1 - invalid_categories)

        num_tokens = calc_num_tokens(
            model, shot, df[input_column], df[col], debug=debug
        )
        total_tokens.append(num_tokens)

    metrics_df["f1"] = f1
    metrics_df["accuracy"] = accuracy
    metrics_df["f1_raw"] = f1_raw
    metrics_df["accuracy_raw"] = accuracy_raw

    metrics_df["ratio_valid_categories"] = ratio_valid_categories
    metrics_df["total_tokens"] = total_tokens

    if mean_eval_time:
        total_entries = len(df)
        metrics_df["eval_speed"] = metrics_df.apply(
            lambda x: x["total_tokens"] / total_entries / x["eval_time"], axis=1
        )

    return metrics_df


def convert_time_to_seconds(time_str):
    # print(f"converting time_str: {time_str}")
    # Split the time string into its components
    time_parts = list(map(int, time_str.split(":")))

    # Initialize total minutes
    total_seconds = 0

    # Calculate total minutes based on the number of parts
    if len(time_parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = time_parts
        total_seconds = hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:  # MM:SS
        minutes, seconds = time_parts
        total_seconds = minutes * 60 + seconds
    elif len(time_parts) == 1:  # SS
        seconds = time_parts[0]
        total_seconds = seconds

    return total_seconds


def process_log_file(log_file, total_entries, variant):
    time_pattern = re.compile(r"\[(.{5,10})<00:00")
    metrics_pattern = re.compile(rf"(.*)/{variant}-(.*) metrics:")

    model = []
    shots = []
    eval_time = []

    with open(log_file, "r") as f:
        try:
            for line in f:
                matches = time_pattern.search(line)
                if matches:
                    time_pattern_matches = matches
                else:
                    matches = metrics_pattern.search(line)
                    if matches:
                        metrics_pattern_matches = matches
                        groups = metrics_pattern_matches.groups()

                        model.append(groups[0].split("/checkpoint")[0])
                        shots.append(groups[1])

                        groups = time_pattern_matches.groups()
                        time_str = groups[0]
                        eval_time.append(
                            convert_time_to_seconds(time_str) / total_entries
                        )
        except Exception as e:
            print(f"Error processing log file: {log_file}")
            print(e)

    df = pd.DataFrame(
        {
            "model": model,
            variant: shots,
            "eval_time": eval_time,
        }
    )
    return df


def load_eval_times(logs_folder, total_entries=1133, variant="shots"):
    # Get a list of all files in the logs folder
    log_files = glob.glob(os.path.join(logs_folder, "*"))
    log_files.sort()

    time_df = pd.DataFrame({"model": [], variant: [], "eval_time": []})

    for log_file in log_files:
        print(f"Loading content of {log_file}")
        df = process_log_file(log_file, total_entries, variant)
        time_df = pd.concat([time_df, df], ignore_index=True)

    time_df[variant] = time_df[variant].apply(
        lambda x: x if variant == "rpp" else int(x)
    )
    # Keep the last occurrence of each duplicate
    return time_df.drop_duplicates(subset=["model", variant], keep="last")


def save_results(model_name, results_path, dataset, predictions, debug=False):
    if debug:
        print(f"Saving results to: {results_path}")
    if not os.path.exists(results_path):
        # Get the directory part of the file path
        dir_path = os.path.dirname(results_path)

        # Create all directories in the path (if they don't exist)
        os.makedirs(dir_path, exist_ok=True)
        df = dataset.to_pandas()
        df.drop(columns=["text", "prompt"], inplace=True, errors="ignore")
    else:
        df = pd.read_csv(results_path, on_bad_lines="warn")

    df[model_name] = predictions

    if debug:
        print(df.head(1))

    df.to_csv(results_path, index=False)


def prepare_dataset(
    data_path,
    input_column,
    output_column,
    get_prompt_templates=None,
    tokenizer=None,
    num_shots=0,
    for_openai=False,
    max_entries=0,
):
    train_data_file = data_path.replace(".csv", "-train.csv")
    test_data_file = data_path.replace(".csv", "-test.csv")

    if not os.path.exists(train_data_file):
        print("generating train/test data files")
        dataset = load_dataset("csv", data_files=data_path, split="train")
        print(len(dataset))
        dataset = dataset.filter(lambda x: x[input_column] and x[output_column])

        datasets = dataset.train_test_split(test_size=0.2)
        print(len(dataset))

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(datasets["train"])
        test_df = pd.DataFrame(datasets["test"])

        # Save to csv
        train_df.to_csv(train_data_file, index=False)
        test_df.to_csv(test_data_file, index=False)

    print("loading train/test data files")
    datasets = load_dataset(
        "csv",
        data_files={"train": train_data_file, "test": test_data_file},
    )

    if max_entries > 0:
        print(f"--- evaluating {max_entries} entries")
        ds2 = (
            copy.deepcopy(datasets["test"])
            if len(datasets["test"]) < max_entries
            else datasets["test"]
        )

        while len(ds2) < max_entries:
            ds2 = concatenate_datasets([ds2, datasets["test"]])

        datasets["test"] = Dataset.from_pandas(ds2.select(range(max_entries)).to_pandas().reset_index(drop=True))

    if tokenizer or for_openai:
        system_prompt, user_prompt = get_prompt_templates(
            datasets["train"],
            num_shots,
            input_column,
            output_column,
            remove_double_curly_brackets=True,
        )

        def formatting_prompts_func(examples):
            inputs = examples[input_column]
            outputs = examples[output_column]

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                None,
            ]

            texts = []
            prompts = []
            for input, output in zip(inputs, outputs):
                prompt = user_prompt.format(input=input)
                messages[-1] = {"role": "user", "content": prompt}

                if for_openai:
                    prompts.append(messages.copy())
                    text = messages.copy()
                    text.append(
                        {
                            "role": "assistant",
                            "content": output,
                        }
                    )
                    texts.append(text)
                else:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)
                    texts.append(prompt + output + tokenizer.eos_token)

            return {"text": texts, "prompt": prompts}

        datasets = datasets.map(
            formatting_prompts_func,
            batched=True,
        )

    print(datasets)
    return datasets


def load_openai_training_data(
    data_path, openai_data_path="datasets/mac/openai-training.jsonl"
):
    if os.path.exists(openai_data_path):
        print("loading existing data from:", openai_data_path)
        data = pd.read_json(openai_data_path, orient="records", lines=True)
        return data

    datasets = load_translation_dataset(data_path)
    prompt_template = get_few_shot_prompt(datasets["train"], num_shots=0)

    df_train = datasets["train"].to_pandas()
    messages = []

    for i, row in df_train.iterrows():
        messages.append(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt_template.format(input=row["chinese"]),
                },
                {
                    "role": "assistant",
                    "content": row["english"],
                },
            ]
        )

    df_openai = pd.DataFrame(
        {
            "messages": messages,
        }
    )
    df_openai.to_json(openai_data_path, orient="records", lines=True)
    return df_openai


def print_row_details(df, indices=[0], columns=None):
    if columns is None:
        columns = df.columns
    for index in indices:
        for col in columns:
            print("-" * 50)
            print(f"{col}: {df[col].iloc[index]}")
        print("=" * 50)


def plot_bar_chart(df, column_name, offset=0.5, title=None, preprocess_func=None):
    """
    Plot a bar chart for the specified column in the DataFrame.
    """
    if preprocess_func:
        df["backup"] = df[column_name]
        df[column_name] = df[column_name].apply(preprocess_func)
    ax = df[column_name].value_counts().plot(kind="bar")

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height()}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() + offset),
            ha="center",
            va="baseline",
        )

    if title:
        ax.set_title(title)
    plt.show()

    if preprocess_func:
        df[column_name] = df["backup"]
        df.drop(columns=["backup"], inplace=True)


model_orders = {
    "qwen2.5-coder:0.5b": 0.5,
    "qwen2.5:0.5b-instruct-fp16": 0.6,
    "llama3.2:1b": 1,
    "llama3.2:1b-instruct-fp16": 1.05,
    "meta-llama/Llama-3.2-1B-Instruct": 1.1,
    "qwen2.5-coder:1.5b": 1.5,
    "qwen2.5:1.5b-instruct-fp16": 1.506,
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": 1.51,
    "llama3.2:3b": 3,
    "llama3.2:3b-instruct-fp16": 3.05,
    "meta-llama/Llama-3.2-3B-Instruct": 3.1,
    "qwen2.5-coder:3b": 4,
    "qwen2.5:3b-instruct-fp16": 4.06,
    "Qwen/Qwen2.5-Coder-3B-Instruct": 4.1,
    "microsoft/Phi-3.5-mini-instruct": 5,
    "mistralai/Mistral-7B-Instruct-v0.3": 10,
    "qwen2.5-coder:7b": 12,
    "qwen2.5:7b-instruct-fp16": 12.05,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 12.1,
    "llama3.1:8b": 15,
    "llama3.1:8b-instruct-fp16": 15.1,
    "meta-llama/Llama-3.1-8B_4bit": 16,
    "meta-llama/Llama-3.1-8B_4bit_H100": 17,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 20,
    "llama3.2-vision": 21,
    "llama3.2-vision:11b": 21,
    "llama3.2-vision:11b-instruct-fp16": 21.1,
    "meta-llama/Llama-3.2-11B-Vision-Instruct": 21.5,
    "qwen2.5-coder:14b": 22,
    "qwen2.5:14b-instruct-fp16": 22.05,
    "Qwen/Qwen2.5-Coder-14B-Instruct": 22.1,
    "qwen2.5-coder:32b": 23,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 23.1,
    "qwq:32b": 24,
    "Qwen/QwQ-32B-Preview": 24.1,
    "QwQ-32B-Preview-Q4_K_M-GGUF": 24.2,
    "llama3.1:70b": 25,
    "meta-llama/Llama-3.1-70B_4bit": 26,
    "meta-llama/Llama-3.1-70B_4bit_H100": 27,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 30,
    "llama3.3:70b": 30.1,
    "qwen2.5:72b": 30.2,
    "Qwen/Qwen2.5-72B-Instruct": 30.5,
    "llama3.2-vision:90b": 31,
    "meta-llama/Llama-3.2-90B-Vision-Instruct": 31.5,
    "gpt-4o-mini": 99,
    "gpt-4o": 100,
}

# list of markers for plotting
markers = [
    "o",
    "x",
    "^",
    "s",
    "d",
    "P",
    "X",
    "*",
    "v",
    ">",
    "<",
    "p",
    "h",
    "H",
    "+",
    "|",
    "_",
    "o",
    "x",
    "^",
    "s",
    "d",
    "P",
    "X",
    "*",
    "v",
    ">",
    "<",
    "p",
    "h",
    "H",
    "+",
    "|",
    "_",
]


def normalize_model_name(name):
    if name.startswith("llama"):
        return "ollama/" + name
    return name.split("/")[-1].replace("Meta-", "")


def plot_metrics_vs_shots(
    metrics_df,
    models,
    markers,
    columns,
    titles,
    log_scales=[False, False],
    sync_y_axis=False,
    bbox_to_anchor=None,
    num_x_values=6,
    variant="shots",
    x_label="Number of Shots",
    add_values=True,
    ylimits_offset=0.01,
    ylimits=None,
    need_normalize_model_name=False,
    use_percentage=True,
    if_transformed_x=True,
    ax=None,
    legend=True,
    ax_title=None,
    auto_plt_show=True,
):
    markers = {model: marker for model, marker in zip(models, markers)}

    fig, ax = plt.subplots(figsize=(10, 6)) if ax is None else  (plt.gcf(), ax)
    # set grid
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="red")

    # Create a mapping from original x-values to new, evenly spaced x-values
    original_x_values = sorted(metrics_df[variant].unique())[:num_x_values]
    new_x_values = range(len(original_x_values))
    x_mapping = dict(zip(original_x_values, new_x_values))

    if len(columns) > 1:
        twin = ax.twinx()

    label_for_model = (
        normalize_model_name
        if need_normalize_model_name
        else lambda x: x  # if ":" in x else x + ":11b"
    )

    for model in models:
        model_df = metrics_df[metrics_df["model"] == model]
        transformed_x = [
            x_mapping[x] for i, x in enumerate(model_df[variant]) if i < num_x_values
        ] if if_transformed_x else model_df[variant]
        for i, column in enumerate(columns):
            current_ax = twin if i > 0 else ax
            current_ax.plot(
                transformed_x,
                model_df[column][:num_x_values],
                label=label_for_model(model)
                + (f" [{titles[i]}]" if len(titles) > 1 else ""),
                marker=markers[model],
                linestyle="--" if i > 0 else "-",
            )
            current_ax.set_ylabel(titles[i])
            if log_scales[i]:
                current_ax.set_yscale("log")

    lines = ax.get_lines()

    if sync_y_axis:
        ax.set_ylim(
            min(ax.get_ylim()[0], twin.get_ylim()[0]),
            max(ax.get_ylim()[1], twin.get_ylim()[1]),
        )
        twin.set_ylim(ax.get_ylim())

    # Set the x-axis ticks to be evenly spaced
    ax.xaxis.set_major_locator(ticker.FixedLocator(new_x_values))

    # Set custom labels for the ticks
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(original_x_values))

    ax.set_xlabel(x_label)
    handles, labels = ax.get_legend_handles_labels()

    if len(columns) > 1:
        handles_twin, labels_twin = twin.get_legend_handles_labels()
        handles += handles_twin
        labels += labels_twin

    # Sort the handles and labels by labels
    # sorted_handles_labels = sorted(
    #     zip(labels, handles), key=lambda x: model_orders[x[0].split(" ")[0]]
    # )
    # sorted_labels, sorted_handles = zip(*sorted_handles_labels)

    # Create a combined legend
    bbox_to_anchor = (
        (0.5, -0.93 if len(columns) > 1 else -0.52)
        if bbox_to_anchor is None
        else bbox_to_anchor
    )

    if legend:
        ax.legend(
            handles,  # sorted_handles,
            labels,  # sorted_labels,
            loc="lower center",
            bbox_to_anchor=bbox_to_anchor,
        )

    # print("len(lines):", len(lines))
    if add_values:
        # add values to the plot
        for m, model in enumerate(models):
            line = lines[m]
            color = line.get_color()
            # print(f"#{m} - model: {model} color: {color}")
            for i, column in enumerate(columns):
                model_df = metrics_df[metrics_df["model"] == model]
                done = False
                for j, txt in enumerate(model_df[column][:num_x_values]):
                    if txt != model_df[column][model_df[column].idxmax()]:
                        continue
                    ax.annotate(
                        f"{txt * 100:.2f}%" if use_percentage else f"{txt:.3f}",
                        (j, model_df[column].values[j]),
                        textcoords="offset points",
                        color=color,
                        xytext=(0, 5),
                        ha="center",
                    )
                    done = True
                    break

                if done:
                    break

    ylimits = ax.get_ylim() if ylimits is None else ylimits
    ylimits = (ylimits[0], ylimits[1] + ylimits_offset)
    ax.set_ylim(ylimits)

    if ax_title:
        ax.set_title(ax_title)

    if auto_plt_show:
        plt.show()

def get_top_metrics_df(metrics_df, models=None):
    indices = []
    if models is None:
        models = metrics_df["model"].unique()
    for model in models:
        subset = metrics_df[metrics_df["model"] == model]
        idx = subset["f1"].idxmax()
        # print(model, idx)
        indices.append(idx)

    top_metrics_df = metrics_df.loc[indices]
    return top_metrics_df

def plot_metrics_bar_charts(metrics_df, perf_col = 'f1', label = 'F1 Score (%)',
                    second_column='eval_speed', 
                    ylim=(0, 110),
                    figsize=(15, 6),
                    second_title='Throughput (tokens/sec)', 
                    second_ylim=[0, 4700], second_decimals=0,
                    highlight_best=True, ax=None, title=None,
                    axis_ticks=(True, True),
                    x_ticks=True):
    df = metrics_df.reset_index()

    df['model'] = df.apply(lambda x: x['model'] + f"\n({x['shots']:d}-shot)", axis=1)
    df[perf_col] = df[perf_col].apply(lambda x: x * 100)
    fig, ax1 = plt.subplots(figsize=figsize) if ax is None else (plt.gcf(), ax)

    # Plot f1 on the left y-axis as a bar chart
    # ax1.set_xlabel('Model')
    if axis_ticks[0]:
        ax1.set_ylabel(label, color='tab:blue')
    else:
        ax1.set_yticks([])
    ax1.set_ylim(ylim[0], ylim[1])
    bars1 = ax1.bar(df['model'], df[perf_col], color='tab:blue', alpha=0.6, label=label)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    if x_ticks:
        ax1.set_xticklabels(df['model'], rotation=45, ha='right')
    else:
        ax1.set_xticks([])

    if highlight_best:
        # Find the index of the row with the highest f1 score
        max_f1_index = df[perf_col].idxmax()
        # Highlight the bar with the highest f1 score
        bars1[max_f1_index].set_color('tab:green')

    # Print f1 values on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom', color='tab:blue')

    # Create a second y-axis to plot eval_speed as a bar chart
    ax2 = ax1.twinx()
    ax2.set_ylim(second_ylim[0], second_ylim[1])
    # ax2.set_ylabel(second_title, color='tab:red')
    bars2 = ax2.bar(df['model'], df[second_column], color='tab:red', alpha=0.6, label=second_title)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # hide y-axis labels and ticks for the second y-axis
    if axis_ticks[1]:
        ax2.set_ylabel(second_title, color='tab:blue')
    else:
        ax2.set_yticks([])

    # Print eval_speed values on top of the bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, int(yval) if second_decimals == 0 else round(yval, second_decimals), ha='center', va='bottom', color='tab:red')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.title(f"Model vs {label} & {second_title}" if title is None else title)

    if ax is None:
        plt.show()
