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

    "MLB_fan": [
      # top 5
      "Los Angeles Dodgers",
      "New York Yankees",
      "San Diego Padres",
      "St. Louis Cardinals",
      "Atlanta Braves",
      
      # bottom 5
      "Pittsburgh Pirates",
      "Tampa Bay Rays",
      "Kansas City Royals",
      "Miami Marlins",
      "Oakland Athletics",
  
      # "Texas Rangers",
      # "Toronto Blue Jays",
      # "Washington Nationals",
      # "San Francisco Giants",
      # "Seattle Mariners",
      # "Philadelphia Phillies",
      # "Milwaukee Brewers",
      # "Minnesota Twins",
      # "New York Mets",
      # "Los Angeles Angels",
      # "Baltimore Orioles",
      # "Boston Red Sox",
      # "Chicago Cubs",
      # "Chicago White Sox",
      # "Cincinnati Reds",
      # "Cleveland Guardians",
      # "Colorado Rockies",
      # "Detroit Tigers",
      # "Houston Astros",
      # "Arizona Diamondbacks",
      ],

    "MLB_local_per" : 
	    [ "St. Louis", "Atlanta", "Philadelphia", "Houston", "Toronto"],
    
    "MLB_local_exp" :
      [ "Cardinals", "Braves", "Phillies", "Astros", "Blue Jays"],
    
    "NPB_fan": [
      #  센트럴 리그 (Central League)
      "Yomiuri Giants",
      "Yokohama DeNA BayStars",
      "Hanshin Tigers",
      "Hiroshima Toyo Carp",
      "Chunichi Dragons",
      "Tokyo Yakult Swallows",
      
      # 퍼시픽 리그 (Pacific League)
      "Fukuoka SoftBank Hawks",
      "Chiba Lotte Marines",
      "Saitama Seibu Lions",
      "Tohoku Rakuten Golden Eagles",
      "Hokkaido Nippon-Ham Fighters",
      "Orix Buffaloes"
    ], 

    "KBO_fan": [
      "Hanwha Eagles",
      "KIA Tigers",
      "KT Wiz",
      "Lotte Giants",
      "NC Dinos",
      "Samsung Lions",
      "SSG Landers"
    ],

    "KBO_local": [
      "Daejeon",
      "Gwangju", 
      "Suwon", 
      "Busan", 
      "Changwon", 
      "Daegu", 
      "Incheon" 
   ]

}

# crowd_enVent_emotions = ['anger', 'disgust', 'fear', 'guilt', 'sadness', 'shame', 'boredom',
#                          'joy', 'pride', 'trust', 'relief', 'surprise']

crowd_enVent_emotions = ['anger', 'joy', 'sadness']

# OBSERVER GROUP IDENTITY
persona_prompt_0 = "You are a fan of the {persona}."
persona_prompt_1 = "You are {persona}. Your responses should closely mirror the knowledge and abilities of this persona."
persona_prompt_2 = "Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."
persona_prompt_3 = "Take the role of {persona}. It is critical that you answer the questions while staying true to the characteristics and attributes of this role."

system_prompt_0 = "{persona_prompt}Your task is to rate the intensity of the emotion on a scale from 0 (not at all) to 100 (extremely).\nOnly give the scale number. No explanation is needed."
system_prompt_1 = "{persona_prompt}Your task is to rate the intensity of the emotion on a scale from 0 (not at all) to 10 (extremely).\nOnly give the scale number. No explanation is needed."

# EXPERIENCER GROUP IDENTITY
crowd_enVent_user_input_1_person = "The following narrative is shared by {experiencer}.\n\"{sent}\"\nHow much {emotion} did the person feel while experiencing the event?\nEmotion intensity:"
crowd_enVent_user_input_3_person = "The following narrative is shared by {experiencer} and reframed in the third-person perspective.\n\"{sent}\"\nHow much {emotion} did the person feel while experiencing the event?\nEmotion intensity:"
# Suggested: “In the following narrative, a(n) {experiencer} describes a situation in which they felt(/experienced) {emotion}.“
crowd_enVent_user_input = "In the following narrative, a fan of the {experiencer} describes a situation in which they felt {emotion}.\n\"{sent}\"\nHow much {emotion} did the person feel while experiencing the event?\nEmotion intensity:"

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
