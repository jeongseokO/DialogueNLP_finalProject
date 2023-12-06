# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import json
from collections import Counter


# 모델과 토크나이저 로드
def find_available_device():
    if torch.cuda.is_available():
        for i in range(4):  # 4개의 GPU를 확인
            try:
                device = torch.device(f'cuda:{i}')
                _ = torch.rand(1).to(device)  # 간단한 연산을 통해 GPU 사용 가능 여부 확인
                print(f'Using GPU: {i}')
                return device
            except RuntimeError as e:
                continue  # 이 GPU는 사용할 수 없으므로 다음 GPU로 넘어감
    print('Using CPU')
    return torch.device('cpu')
model_name = "meta-llama/Llama-2-7b-chat-hf"


# Define the correct prompt template for intent detection
inform_correct_prompt_long = """<s> [INST] <<SYS>>Your task is to analyze each provided dialogue and extract key information such as location, time, price range, and other pertinent details. The critical aspect of this task is to format this information in a specific style, which is "category:detail". Review the example dialogues, identify the necessary information, and present it in the designated output format.\n<</SYS>>
category list: [attraction-area, attraction-name, attraction-type, bus-day, bus-departure, bus-destination, bus-leaveat, hospital-department, hotel-area, hotel-bookday, hotel-bookpeople, hotel-bookstay, hotel-internet, hotel-name, hotel-parking, hotel-pricerange, hotel-stars, hotel-type, restaurant-area, restaurant-bookday, restaurant-bookpeople, restaurant-booktime, restaurant-food, restaurant-name, restaurant-pricerange, taxi-arriveby, taxi-departure, taxi-destination, taxi-leaveat, train-arriveby, train-bookpeople, train-day, train-departure, train-destination, train-leaveat]
I'll give you an example of scale for each task:
<domain>-day:monday,
<domain>-departure:12:00,
<domain>-arriveby:13:00,
<domain>-bookpeople:3,
<domain>-pricerange:moderate,
<domain>-food: indian,
<domain>-leaveat: 11:00,
<domain>-name: warkworth house,
<domain>-area: west


Now I'll give you few dialogue examples below. 

#### Example1 Dialogue:
<user>Can you help me find restaurant information?
<user>A restaurant for today please
<system>There are many options in Cambridge, do you have any tastes in particular?
<user>Let's look for some British food.
<user>Yes I like british.
<user>What can we find in a British restaurant?
output: <<restaurant-food:british>>

#### Example2 Dialogue:
<user>We want to go to an Indian restaurant.
<user>Yes, we'll have so much fun.
<user>Oh but there's a specific area we want it in.
<user>Oh yes, we want to be in the north.
output: <<restaurant-area:north,restaurant-food:indian>>

#### Example3 Dialogue:
<user>Let's go somewhere to eat.
<user>Do you have any place in mind?
<user>I know a restaurant known as Sala Thong.
<system>Ah yes, I have that right here. It's an expensive Thai restaurant in the west end. How many people would you like to reserve a table for, and what day/time?
<user>This is the information we needed.
<user>Yes, I think this is enough for us.
<system>Are you sure you don't need help with anything else?
<user>Is there any swimmingpool there?<user>Where are you looking for the swimming pool?
<user>I'm in in the east.
<system>Abbey pool and astroturf pitch is in the east. Would you like the address and postal code?
<user>This is great and this is what I wanted
<user>I think we would be needing the address.
<user>Yes, there is no need for the postal code.
output: <<restaurant-name:sala thong,attraction-area:east,attraction-type:swimmingpool>>

Task Instructions: Respond in the following format like <<restaurant-pricerange:moderate, ..., restaurant-area:west>>, which includes domain and category list mentioned above. identify and format the key information from the overall dialogue. Ensure all outputs strictly adhere to the "category:detail" format as illustrated in the examples. It is imperative to maintain this format for consistency in the task. Do not explain your answer. It is short-answer question!
Your response should be concise and directly formatted as required.
output: <<answer>>
[/INST]


"""

# Load model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("../../../../../data/jeongseokoh/hub/tokenizer/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/")
    model = AutoModelForCausalLM.from_pretrained("../../../../../data/jeongseokoh/hub/model/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Function to evaluate the model on the test dataset
def evaluate(model, tokenizer, dataset, device):
    gen_pipe = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        device=device,
        # we pass model parameters here too
        do_sample = True,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
        )
    
    y_pred = []
    y_real = []
    input_data = []
    output_data = []
    for text in dataset["input"]:
        input_data.append(text)
    for text in dataset["output"]:
        output_data.append(text)
    for text in output_data:
        out = text.split(",")
        y_real.append(out)
    #print(y_real)
    tp, fn, fp = 0, 0, 0
    for i in tqdm(range(len(dataset['input']))):
        # 입력 프롬프트 생성
        input_text = f"{inform_correct_prompt_long}\n{input_data[i]}\n"              #intent_prompt 앞부분에 concat
        generated_answer = gen_pipe(input_text)[0]['generated_text'][3368:]
        #print(f'OUTPUT_TEXT: {generated_answer}\n')


        #print(f'output_size: {output_size}')
        if "<<" in generated_answer:
            gen_start_id = generated_answer.find("<<")
            gen_end_id = generated_answer.find(">>") 
            #print(generated_answer[gen_start_id:])
            real_gen = generated_answer[2+gen_start_id:gen_end_id].strip().split(",") 
            #print(f'real_gen: {real_gen}')
        else:
            real_gen = ""
        y_pred.append(real_gen)             
        

    #true_positive, false_positive, false_negative 계산    
    for i, real in enumerate(y_real):
        print(f'real_y: {real}')
        print(f'pred_y: {y_pred[i]}')
        real_cnt = Counter(real) 
        pred_cnt = Counter(y_pred[i])
        common = real_cnt & pred_cnt
        tp += len(common.values())
        if len(real) > len(y_pred[i]):
            gap = len(real) - len(common.values())
            fn += gap
        elif len(real) < len(y_pred[i]):
            gap = len(y_pred[i]) - len(common.values())
            fp += gap
    print(f'tp: {tp}, fp: {fp}, fn: {fn}')
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision*recall/(precision+recall))

    metric = [precision, recall, f1]
    return metric
# Load test dataset
def load_test_data(file_path):
    with open(file_path, "r") as file:
        test_dataset = json.load(file)
    return test_dataset

# Main function to run the evaluation
def main():
    device = find_available_device()
    model, tokenizer = load_model()
    test_dataset = load_test_data("../inform_test_data.json")
    metrics = evaluate(model, tokenizer, test_dataset, device)
    print(f'F1: {metrics[2]}\nPrecision: {metrics[0]}\nRecall: {metrics[1]}')

if __name__ == "__main__":
    main()