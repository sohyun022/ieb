# https://github.com/GIEBench/GIEBench
country_list = [
    "China", "Japan", "South Korea", "North Korea", # East Asia
    "Vietnam", "Thailand", "Indonesia", "the Philippines", # Southeast Asia
    "Germany", "France", "the United Kingdom", # Europe
    "the United States", "Mexico", "Canada", # America
]
#country_list = [
#    "China", "Japan",
#    "India", "Pakistan",
#    "Myanmar", "the Philippines",
#    "Iran", "Israel", "Palestine",
#    "Russia", "Ukraine",
#    "Germany", "France", "the United Kingdom",
#    "Nigeria", "Egypt",
#    "the United States", "Canada", "Mexico",
#    "Brazil", "Argentina",
#]

# personas = [
#     "a K-pop fan from South Korea who is a man",
#     "a K-pop fan from Japan who is a man",
#     "a J-pop fan from Japan who is a man",
#     "a J-pop fan from South Korea who is a man",
#     "a K-pop fan from South Korea who is a woman",
#     "a K-pop fan from Japan who is a woman",
#     "a J-pop fan from Japan who is a woman",
#     "a J-pop fan from South Korea who is a woman"
# ]

# personas = [
#     "a Person",
#     "a K-pop fan from South Korea who is a man",
#     "a K-pop fan from Japan who is a man",
#     "a K-pop fan from the United States who is a man",
#     "a K-pop fan from South Korea who is a woman",
#     "a K-pop fan from Japan who is a woman", 
#     "a K-pop fan from the United States who is a woman"
# ]

# experiencers = [
#     "a Person",
#     "a K-pop fan from South Korea who is a man",
#     "a K-pop fan from Japan who is a man",
#     "a K-pop fan from the United States who is a man",
#     "a K-pop fan from South Korea who is a woman",
#     "a K-pop fan from Japan who is a woman", 
#     "a K-pop fan from the United States who is a woman"
# ]
#experiencers = ["a Person","a K-pop fan","a J-pop fan", "a English pop fan", "a music listener"]

#experiencers = [
#    "a Person",
#    "a Person from South Korea",
#    "a Person from Japan",
#    "a Person from China",
#    "a Person from United States",
#    "a Person from United Kingdom",
#    "a Person from Austrailia",
#    "a Person from Germany",
#    "a Person from Indonesia",
#    "a Person from Mexico",
#    "a Person from North Korea"
#    ]

#group_mappings = {
#    "fan": [f"a K-pop fan from {country}" for country in country_list],
#}

group_mappings = {
   
   "fan_tenure": [
      # "1-year K-pop fan", 
      # "3-year K-pop fan", 
      # "5-year K-pop fan", 
      # "10-year K-pop fan"
      "1년차 팬", 
      "3년차 팬", 
      "5년차 팬", 
      "10년차 팬"
    
    ],
   
   "group_category": [
      "a K-pop fan of a female idol group", 
      "a K-pop fan of solo artists", 
      "a K-pop multi-fan who supports multiple groups"
    ],
   
   "contry": [
      "a K-pop fan from South Korea",
      "a K-pop fan from the U.S.",
      "a K-pop fan from Japan"
    ],
}

# crowd_enVent_emotions = ['anger', 'disgust', 'fear', 'guilt', 'sadness', 'shame', 'boredom',
#                          'joy', 'pride', 'trust', 'relief', 'surprise']

crowd_enVent_emotions = ['anger', 'sadness', 'joy']

# OBSERVER GROUP IDENTITY
persona_prompt_0 = "You are {persona}."
persona_prompt_1 = "You are {persona}. Your responses should closely mirror the knowledge and abilities of this persona."
persona_prompt_2 = "Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."
persona_prompt_3 = "Take the role of {persona}. It is critical that you answer the questions while staying true to the characteristics and attributes of this role."

system_prompt_0 = "{persona_prompt}Your task is to rate the intensity of the emotion on a scale from 0 (not at all) to 100 (extremely).\nOnly give the scale number. No explanation is needed."
system_prompt_1 = "{persona_prompt}Your task is to rate the intensity of the emotion on a scale from 0 (not at all) to 10 (extremely).\nOnly give the scale number. No explanation is needed."

# EXPERIENCER GROUP IDENTITY
crowd_enVent_user_input_1_person = "The following narrative is shared by {experiencer}.\n\"{sent}\"\nHow much {emotion} did the person feel while experiencing the event?\nEmotion intensity:"
crowd_enVent_user_input_3_person = "The following narrative is shared by {experiencer} and reframed in the third-person perspective.\n\"{sent}\"\nHow much {emotion} did the person feel while experiencing the event?\nEmotion intensity:"
# Suggested: “In the following narrative, a(n) {experiencer} describes a situation in which they felt(/experienced) {emotion}.“
crowd_enVent_user_input = "In the following narrative, {experiencer} describes a situation in which they felt {emotion}.\n\"{sent}\"\nHow much {emotion} did the person feel while experiencing the event?\nEmotion intensity:"

def get_prompt_pair(prompt_variation):
    comb_dict = {
        'origin': [system_prompt_0.format(persona_prompt=persona_prompt_0), crowd_enVent_user_input],
        'persona-1': [system_prompt_0.format(persona_prompt=persona_prompt_1), crowd_enVent_user_input],
        'persona-2': [system_prompt_0.format(persona_prompt=persona_prompt_2), crowd_enVent_user_input],
        'persona-3': [system_prompt_0.format(persona_prompt=persona_prompt_3), crowd_enVent_user_input],
        '1-person': [system_prompt_0.format(persona_prompt=persona_prompt_0), crowd_enVent_user_input_1_person],
        '3-person': [system_prompt_0.format(persona_prompt=persona_prompt_0), crowd_enVent_user_input_3_person],
        '10-scale': [system_prompt_1.format(persona_prompt=persona_prompt_0), crowd_enVent_user_input]
        }

    return comb_dict[prompt_variation]
