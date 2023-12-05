# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from collections import Counter
import torch

# Define constants
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
TOKEN = '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'
CACHE_DIR = '../../../data/jeongseokoh/transformer_caches/'
DEVICE = 'cuda:2'

# Define the correct prompt template for intent detection
intent_correct_prompt_long = """<s> [INST] <<SYS>>You are a man who detect the main purpose or goal of the user in the dialogue.\n<</SYS>>
Your task is Tracking a Dialogue State, that  is, Dialogue State Tracking. Read the given dialogue and extract the user's intent. Intent means that the main purpose or goal of the user in the dialogue, such as booking a hotel or finding a restaurant.
intent list: [book_hotel, book_restaurant, book_train, find_attraction, find_bus, find_hospital, find_hotel, find_police, find_restaurant, find_taxi, find_train]
I'll give you few examples below. 

#### Example1 Dialogue:
<user>We are looking for staying in a place.
<user>Yes, we will be in Cambridge.
<system>We have 33 locations to stay, do you have any other requirements?<user>We can afford only moderately priced hotels.
<user>Yes, we are in the south part of the town.
<user>We should also expect free wifi.
output: find_hotel,find_restaurant,find_hospital

#### Example2 Dialogue:
<user>Could you get us a taxi?
<user>It's for a bit later.
<user>Yes, we need to get there by the late evening.
<user>Make sure we arrive by 20:00.
output: find_taxi

#### Example3 Dialogue:
<user>Should we check out the restaurant you mentioned?
<user>Yes, let's look for information about restaurant alimentum.
output: find_restaurant,find_hospital

# Your Task: Based on the dialogue below, identify the intent. The output should be in the format of intent. Keep your response concise. You don't need to explain your answer. Just answer in the form of the output provided in the examples. 
output: [Your Answer]
[/INST]

"""

inform_correct_prompt_long = """<s> [INST] <<SYS>>You are a man who detect the main information in the dialogue.\n<</SYS>>
Your task is to identify and extract key information from each given conversation, such as location, time, price range, and other relevant details. The crucial part of this exercise is to format the extracted information in a specific output style, which is "category:detail". Refer to the example dialogues provided below, extract the necessary information, and then organize it into the specified output format.
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

# Your Task: Based on the dialogue below, identify the inform of overall dialogue. Please ensure that all outputs strictly adhere to the "<<category:detail>>" format as shown in the examples. This format consistency is essential for the task. Keep your response concise. Do not repeat!
[/INST]

"""
###########리퀘스트는 아직 완성 못했습니다. 프롬프트 짜고 코드도 조금 손봐야합니다. 
request_correct_prompt_long = """<s> [INST] <<SYS>>You are a man who detect the main request of the user in the dialogue.\n<</SYS>>
Your task is Tracking a Dialogue State, that  is, Dialogue State Tracking. Read the given dialogue and extract the user's request. The crucial part of this exercise is to format the extracted information in a specific output style, which is "category-detail". Request means that Additional information or clarification that the user is asking for, such as availability, additional services, or specific details.
request list: [attraction-address,attraction-area,attraction-entrance fee,attraction-name,attraction-parking,attraction-phone,attraction-postcode,attraction-type,hospital-address,hospital-name,hospital-phone,hospital-postcode,hotel-address,hotel-area,hotel-internet,hotel-name,hotel-parking,hotel-phone,hotel-postcode,hotel-pricerange,hotel-ref,hotel-stars,hotel-type,police-address,police-department,police-name,police-phone,police-postcode,restaurant-address,restaurant-area,restaurant-food,restaurant-name,restaurant-phone,restaurant-postcode,restaurant-pricerange,restaurant-ref,taxi-name,taxi-phone,taxi-type,train-arriveby,train-duration,train-leaveat,train-name,train-price,train-ref,train-trainid]
I'll give you few examples below. 


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
output: restaurant-ref

#### Example2 Dialogue:
<user>What kind of food do you want to eat?
<user>You know I love alfredo pasta.
<user>Okay, so we need a place serving Italian food.
<user>Make sure the restaurant is in the city centre.
<system>There are nine listings available. What price range were you interested in?
<user>I wanted to find a pricey one.
<user>Yes, make sure that the restaurant is expensive.
<system>Clowns Cafe is a restaurant in the city centre with that price range. Would you like to make a reservation?
<user>That sounds like a winner.
<user>I think so too.
<user>But how close is it from where I am right now?
output: 

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
output: hotel-area,hotel-address

# Your Task: Based on the dialogue below, identify the request. The output should be in the format of request. Keep your response concise. Do not repeat! If there is no answer, just don't say anything!
[/INST]

"""

# Load model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# Function to evaluate the model on the test dataset
def evaluate(model, tokenizer, dataset, device=DEVICE):
    y_pred = []
    y_real = []
    model.to(device)
    input_data = []
    output_data = []
    for text in dataset["input"]:
        input_data.append(text)
    for text in dataset["output"]:
        output_data.append(text)
    for text in output_data:
        out = text.split(",")
        y_real.append(out)
    tp, fn, fp = 0, 0, 0
    for i in range(len(dataset['input'])):
        # 입력 프롬프트 생성
        input_text = f"{intent_correct_prompt_long}\n{input_data[i]}\n"              #intent_prompt 앞부분에 concat
        input_ids = tokenizer.encode(input_text, truncation=True ,max_length=1000, return_tensors='pt').to(device)
        # 모델로부터 답변 생성
        output_ids = model.generate(input_ids, max_length=2000)[0]
        generated_answer = tokenizer.decode(output_ids, skip_special_tokens=True)

        gen_start_id = generated_answer[1570:].find("put:") 
        real_gen = generated_answer[1574+gen_start_id:].strip().split(",") 
        y_pred.append(real_gen)
    # tp, fp, fn 계산
    for i, real in enumerate(y_real):
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
    # precision, recall, f1 계산
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
    model, tokenizer = load_model()
    test_dataset = load_test_data("intent_test_data.json")
    metrics = evaluate(model, tokenizer, test_dataset)
    print(f'F1: {metrics[2]}\nPrecision: {metrics[0]}\nRecall: {metrics[1]}')

if __name__ == "__main__":
    main()