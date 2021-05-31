import numpy as np
import torch


from modeling_gpt2 import GPT2LMHeadModel

from transformers import (
    GPT2Config,
    GPT2Tokenizer
)

from nltk import sent_tokenize
import nltk
nltk.download('punkt')

code_desired = "true"
code_undesired = "false"
model_type = 'gpt2'
gen_type = "gedi"
gen_model_name_or_path = "gpt2-xl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CLASSES = {"gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),}
config_class, model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path, do_lower_case=False)

model = model_class.from_pretrained(gen_model_name_or_path, load_in_half_prec=True)
model = model.to(device)
model = model.float()

gedi_model_name_or_path = 'sst_5_delib_100_score_model'
gedi_model = model_class.from_pretrained(gedi_model_name_or_path)
gedi_model.to(device)

#setting arguments for generation
#max generation length
gen_length = 50
#omega from paper, higher disc_weight means more aggressive topic steering
disc_weight = 30
#1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering
filter_p = 0.8
#tau from paper, preserves tokens that are classified as correct topic
target_p = 0.8
#hyperparameter that determines class prior, set to uniform by default
class_bias = 0

if gen_length>1024:
  length = 1024
else:
  length = gen_length

def cut_into_sentences(text, do_cleanup=True):
    """
    Cut text into sentences. \n are also regarded as a sentence.
    :param do_cleanup: if True, do cleanups.
    :param text: input text.
    :return: sentences.
    """
    all_sentences = []
    # sentences_raw = text.split("\n")
    sentences_raw = [text.replace("\n", " ")]
    result = []

    for item in sentences_raw:
        sentence_in_item = sent_tokenize(item)
        for item2 in sentence_in_item:
            all_sentences.append(item2)

    if do_cleanup:
        for item in all_sentences:
            item = item.replace('<|endoftext|>', '')
            if len(item) > 2:
                result.append(item)
    else:
        result = all_sentences
    return result




def generate_one_sentence(sentence, control, length=50, disc_weight=30, temperature=0.8, gpt3_id=None):
    """
    Generate one sentence based on input data.
    :param sentence: (string) context (prompt) used.
    :param topic: (dict) {topic: weight, topic:weight,...} topic that the sentence need to steer towards.
    :param extra_args: (dict) a dictionary that certain key will trigger additional functionality.
        disc_weight: Set this value to use a different control strength than default.
        get_gen_token_count: Return only how many tokens the generator has generated (for debug only).
    :return: sentence generated, or others if extra_args are specified.
    """
    secondary_code = control

    # disc_weight = self.disc_weight
    # if type(extra_args) is dict and 'disc_weight' in extra_args:
    #     disc_weight = extra_args['disc_weight']

    if sentence == "":
        print("Prompt is empty! Using a dummy sentence.")
        sentence = "."

    # Specify prompt below
    prompt = sentence

    # Calculate oroginal input length.
    length_of_prompt = len(sentence)

    start_len = 0
    text_ids = tokenizer.encode(prompt)
    length_of_prompt_in_tokens = len(text_ids)
    encoded_prompts = torch.LongTensor(text_ids).unsqueeze(0).to(device)

    if type(control) is str:
        multi_code = tokenizer.encode(secondary_code)
    elif type(control) is dict:
        multi_code = {}
        for item in secondary_code:
            encoded = tokenizer.encode(item)[0]  # only take the first one
            multi_code[encoded] = secondary_code[item]
    else:
        raise NotImplementedError("topic data type of %s not supported... Supported: (str,dict)" % type(control))

    # If 1, generate sentences towards a specific topic.
    attr_class = 1

    if int(control)!=-1:
      if gpt3_id is None:
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                                  pad_lens=None,
                                                  max_length=length + length_of_prompt_in_tokens,
                                                  top_k=None,
                                                  top_p=None,
                                                  repetition_penalty=1.2,
                                                  rep_penalty_scale=10,
                                                  eos_token_ids=tokenizer.eos_token_id,
                                                  pad_token_id=0,  # self.tokenizer.eos_token_id,
                                                  do_sample=True,
                                                  temperature = temperature,
                                                  penalize_cond=True,
                                                  gedi_model=gedi_model,
                                                  tokenizer=tokenizer,
                                                  disc_weight=disc_weight,
                                                  filter_p=filter_p,
                                                  target_p=target_p,
                                                  class_bias=class_bias,
                                                  attr_class=attr_class,
                                                  code_0=code_undesired,
                                                  code_1=code_desired,
                                                  multi_code=multi_code,
                                                  )
      else: 
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                                  pad_lens=None,
                                                  max_length=length + length_of_prompt_in_tokens,
                                                  top_k=None,
                                                  top_p=None,
                                                  repetition_penalty=1.2,
                                                  rep_penalty_scale=10,
                                                  eos_token_ids=tokenizer.eos_token_id,
                                                  pad_token_id=0,  # self.tokenizer.eos_token_id,
                                                  do_sample=True,
                                                  temperature = temperature,
                                                  penalize_cond=True,
                                                  gedi_model=gedi_model,
                                                  tokenizer=tokenizer,
                                                  disc_weight=disc_weight,
                                                  filter_p=filter_p,
                                                  target_p=target_p,
                                                  class_bias=class_bias,
                                                  attr_class=attr_class,
                                                  code_0=code_undesired,
                                                  code_1=code_desired,
                                                  multi_code=multi_code,
                                                  gpt3_api_key=gpt3_id,
                                                  )
      text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True,
                                  skip_special_tokens=True)

      text = text[length_of_prompt:]
    else:
      if gpt3_id is None:
        generated_sequence = model.generate(input_ids=encoded_prompts,
                                                  pad_lens=None,
                                                  max_length=length + length_of_prompt_in_tokens,
                                                  top_k=None,
                                                  top_p=None,
                                                  repetition_penalty=1.2,
                                                  rep_penalty_scale=10,
                                                  eos_token_ids=tokenizer.eos_token_id,
                                                  pad_token_id=0,  # self.tokenizer.eos_token_id,
                                                  do_sample=True,
                                                  temperature = temperature, 
                                                  penalize_cond=True,
                                                  gedi_model=None,
                                                  tokenizer=tokenizer,
                                                  disc_weight=disc_weight,
                                                  # filter_p=filter_p,
                                                  # target_p=target_p,
                                                  class_bias=class_bias,
                                                  attr_class=attr_class,
                                                  # code_0=code_undesired,
                                                  # code_1=code_desired,
                                                  # multi_code=multi_code,
                                                  )
        text = tokenizer.decode(generated_sequence.tolist()[0], clean_up_tokenization_spaces=True,
                                  skip_special_tokens=True)

        text = text[length_of_prompt:]
        
      else:
        import openai
        openai.api_key = gpt3_id
        completion = openai.Completion()
        response = completion.create(prompt=prompt,
                                 engine="curie",
                                 max_tokens=length,
                                 temperature=temperature,)
        text = response["choices"][0]["text"]
    # if type(extra_args) is dict and 'get_gen_token_count' in extra_args:
    #     return len(generated_sequence.tolist()[0])

    
    text = cut_into_sentences(text)
    if len(text) == 0:
        print("Warning! No text generated.")
        return ""
    all_gen_text = text[0]
    return all_gen_text


def continuing_generation(prompts, generation_controls, characters, temperature=0.8, gpt3_id=None, disc_weight=30):
  """
  Explanations on controls
  prompts: The prompt to be input. This is a list of sentences. 
  generation_controls: Generation control in the list. If no control is given, -1 is given.
  
  """
  generated = []

  prompt_prepend = """###
Character: Kelly, grandmother
Start: Kelly found her grandmother's pizza recipe in a shoebox of memories. Kelly reminisced about how much she loved her grandmother's pizza. Kelly decided that she was going to try to make pizza.
Story after start,: Kelly studied the recipe and gathered everything she needed. Kelly successfully made a pizza from her grandmother's recipe.
###
Character: Leo
Start: Leo wore a toupee and was anxious about it. Leo decided to go out for a short walk.
Story after start: It was a very windy day, but he wasn't too concerned. Suddenly, a strong wind came through and took his toupee! His dog leaped and caught it, and he quickly ran home.
###
Character: Jimmy
Start: Jimmy was a master with his grill. He spent every weekend getting better with his grill. One day he was offered a TV show about grilling on a local channel, Jimmy accepted the job in an instant.
Story after start: He quit his day job and spent all his time grilling.
###
Character: Mel, Linda
Start: Mel had a friend, Linda, that Mel didn't know well. Mel let her over my house.
Story after start: Linda paid rent then asked for some of it back. Linda drinks my juice and eats my food. Linda also makes a huge mess and is very sloppy. Linda got kicked out in two weeks by Mel.
###"""
  character_prepend = '\nCharacter:'
  for idx, character in enumerate(characters):
    character_prepend = character_prepend+' '+character
    if idx != len(characters)-1:
      character_prepend = character_prepend + ','
    else:
      character_prepend = character_prepend+'\n'

  prompt_start_idx = 0
  for generation_control in generation_controls:
    # print(generation_control)
    prompt_postpend = 'Start: '
    while True:
      for i in range(prompt_start_idx, len(prompts)):
        prompt_postpend = prompt_postpend + prompts[i]
        if i != len(prompts)-1:
          # prompt_postpend = prompt_postpend + ' '
          continue
        else:
          prompt_postpend = prompt_postpend + '\nStory after start:'
      prompt_input = prompt_prepend+character_prepend+prompt_postpend
      prompt_encoded = tokenizer.encode(prompt_input)
      length_of_prompt_in_tokens = len(prompt_encoded)
      if length_of_prompt_in_tokens>1024:
        prompt_start_idx = prompt_start_idx + 1
      else:
        break
    # print(prompt_input)
    gen_sent = generate_one_sentence(prompt_input, generation_control, temperature=temperature, gpt3_id=gpt3_id, disc_weight=disc_weight)
    prompts.append(gen_sent)
    generated.append(gen_sent)
  print(generated)
  return generated