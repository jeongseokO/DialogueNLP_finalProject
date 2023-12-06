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


# Define the correct prompt template for request detection
request_correct_prompt_long = """<s> [INST] <<SYS>>As a language model, your role is to determine the primary purpose or goal of the user in a given dialogue. This task is known as Dialogue State Tracking. Your specific task is to read the provided dialogue and identify the user's request. "Request" refers to the additional information or clarification that the user is asking for, such as availability, additional services, or specific details. \n<</SYS>>
Make a response referring to request list: [attraction-address,attraction-area,attraction-entrance fee,attraction-name,attraction-parking,attraction-phone,attraction-postcode,attraction-type,hospital-address,hospital-name,hospital-phone,hospital-postcode,hotel-address,hotel-area,hotel-internet,hotel-name,hotel-parking,hotel-phone,hotel-postcode,hotel-pricerange,hotel-ref,hotel-stars,hotel-type,police-address,police-department,police-name,police-phone,police-postcode,restaurant-address,restaurant-area,restaurant-food,restaurant-name,restaurant-phone,restaurant-postcode,restaurant-pricerange,restaurant-ref,taxi-name,taxi-phone,taxi-type,train-arriveby,train-duration,train-leaveat,train-name,train-price,train-ref,train-trainid]
I'll give you few examples below. 

<</SYS>>
#### Example1 Dialogue:
<user>Let's go somewhere to eat. I just want to eat Chinese?
<user>I will join you. Can you suggest us some Chinese restaurants that are located in the center of the town?
<system>There are 10 Chinese restaurants in the centre of town. Would you like a moderately priced one or an expensive one? We have a few cheap.
<user>I don't want to go with a cheap one. What do you save my buddy?
<user>I also agree with you on this. I think we should avoid the cheap one.
<user>So it is final that we are going ahead with an expensive one.
<system>I have found tang Chinese which is here in the centre. Would you like to start a reservation?
<user>If it fulfills our criteria, I think we should go ahead with the booking. What do you say on this?
<user>I think we should go ahead with the booking. I forgot to tell you that two of my friends are also joining us.
<user>So that would make us a total of 4 people. Is Friday fine for you?
<user>Yes, Friday is good for us. So, please make a reservation for four people at 16:15 on Friday. Please save the reference number at the earliest.
Request: restaurant-ref

#### Example2 Dialogue:
<user>we are looking for a place to go
<user>yes, an attraction in cambridge
<user>perhaps an architecture attraction
<system>There are 5 architecture attractions in the centre. WIll one of those work?
<user>i'm sure one will work
<user>what would you suggest?
<user>yes, could you please recommend one for us
Request: attraction-name

#### Example3 Dialogue:
<user>We are going to Cambridge.
<user>Yes, we want to try the local restaurants there.
<user>But we will also need a place to stay.
<user>Yes we would need a hotel.
<system>There are many restaurants. Can you please elaborate on what you would like?
<user>First, we will need a hotel with a particular rating.
<user>Yes, we are looking for something 3 stars.
<user>We would need internet as well.
<user>Yes, so we need to make sure that it has free wifi.
<system>I have 5 options for you, located all over town. Do you have a certain area or price range in mind?
<user>We would like some cheap price range place with chinese food.
<user>Yes, we will be in the west part of the town.
<user>We would need an expensive hotel with 3 stars and wifi.
<user>Yes, we would need the hotel address, area and postcode as well.
Request: hotel-area,hotel-address

Task Instructions: Identify the user's request. Your response should be concise and formatted as the request only. Do not explain your answer. It is a short-answer question. There is no need for an explanatory response. Simply provide your answer in the following format:
Request: Your Answer
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
        input_text = f"{request_correct_prompt_long}\n{input_data[i]}\n"              #intent_prompt 앞부분에 concat
        generated_answer = gen_pipe(input_text)[0]['generated_text'][3860:]
        #print(f'OUTPUT_TEXT: {generated_answer}\n')

        
        #print(f'output_size: {output_size}')
        if "put:" in generated_answer:
            gen_start_id = generated_answer.find("put:")
            print(generated_answer[gen_start_id:])
            real_gen = generated_answer[4+gen_start_id:].replace("*","").replace("\n","").strip().split(",") 
            #print(f'real_gen: {real_gen}')
        elif "est:" in generated_answer:
            gen_start_id = generated_answer.find("est:")
            #print(generated_answer[gen_start_id:])
            real_gen = generated_answer[4+gen_start_id:].replace("*","").replace("\n","").strip().split(",") 
        else:
            real_gen = ["none"]
        y_pred.append(real_gen)             
        

    #true_positive, false_positive, false_negative 계산    
    for i, real in enumerate(y_real):
        #print(f'real_y: {real}')
        #print(f'pred_y: {y_pred[i]}')
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
    test_dataset = load_test_data("../request_test_data.json")
    metrics = evaluate(model, tokenizer, test_dataset, device)
    print(f'F1: {metrics[2]}\nPrecision: {metrics[0]}\nRecall: {metrics[1]}')

if __name__ == "__main__":
    main()