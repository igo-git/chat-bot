from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria
import torch
import sys
from prompting import build_prompt_for
import yaml
from yaml.loader import SafeLoader

class _SentinelTokenStoppingCriteria(StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False

def initializeModel(model_name):
    model_sub_name = model_name.split('/')[1]
    if model_sub_name.startswith('blenderbot'):
        print('Initialization of ' + model_name + ' bot.')
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        print('Bot ' + model_name + ' initialized.')
        return model, tokenizer
    else:
        print('Initialization of ' + model_name + ' bot.')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print('Bot ' + model_name + ' initialized.')
        return model, tokenizer
    return None, None

class AnnBot():
    def __init__(self, config_file_name='config.yaml'):
        self.first_step = True
        self.config_file_name = config_file_name
        try:
            with open(self.config_file_name) as file:
                config = yaml.load(file, Loader=SafeLoader)
        except Exception:
            print('Cann\'t read config file')
            exit(0)
        self.history = []
        if 'char_name' in config['chat_bot'].keys():
            self.char_name = config['chat_bot']['char_name']
        else:
            self.char_name = 'Bot'
        if 'bot_language' in config['chat_bot'].keys():
            self.bot_language = config['chat_bot']['bot_language']
        else:
            self.bot_language = 'en'
        if 'char_persona_file_name' in config['chat_bot'].keys():
            self.char_persona_file_name = config['chat_bot']['char_persona_file_name']
        else:
            self.char_persona_file_name = None
        if 'example_dialogue_file_name' in config['chat_bot'].keys():
            self.example_dialogue_file_name = config['chat_bot']['example_dialogue_file_name']
        else:
            self.example_dialogue_file_name = None
        if 'history_file_name' in config['chat_bot'].keys():
            self.history_file_name = config['chat_bot']['history_file_name']
        else:
            self.history_file_name = None
        self.load_config_data()
        if 'model_name' in config['chat_bot'].keys(): 
            self.model_name = config['chat_bot']['model_name']
        else:
            print('No model name specified in config file')
            exit(0)
        self.model, self.tokenizer = initializeModel(self.model_name)
        if self.model is None:
            print('No model initialized')
            sys.exit(0)
        if 'need_translation' in config['chat_bot'].keys():
            self.need_translation = config['chat_bot']['need_translation']
        else:
            self.need_translation = False
        if 'need_voice' in config['chat_bot'].keys():
            self.need_voice = config['chat_bot']['need_voice']
        else:
            self.need_voice = False
        model_sub_name = self.model_name.split('/')[1]
        if model_sub_name.startswith('pygmalion'):
            self.getAnswer = self.getAnswerPygmalion
        elif model_sub_name.startswith('ruDialoGPT'):
            self.getAnswer = self.getAnswerRuDialoGPT
        elif model_sub_name.startswith('blenderbot'):
            self.getAnswer = self.getAnswerBlenderbot

    def getAnswerBlenderbot(self, message):
        inputs = self.tokenizer([message], return_tensors='pt')
        reply_ids = self.model.generate(**inputs, max_new_tokens=40)
        answer = self.char_name + ': ' + self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return answer

    def getAnswerBlenderbotWithHistory(self, message):
        user_input_ids = self.tokenizer(message + self.tokenizer.eos_token, return_tensors='pt')
        if self.first_step:
            self.first_step = False
            bot_input_ids = user_input_ids
        else:
            bot_input_ids = torch.cat([self.chat_history_ids, user_input_ids], dim=-1)
        self.chat_history_ids = self.model.generate(**bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
#        answer = self.char_name + ': ' + self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        answer = self.char_name + ': ' + self.tokenizer.batch_decode(self.chat_history_ids, skip_special_tokens=True)[0]
        return answer

    def getAnswerRuDialoGPT(self, message):
        if len(self.history) < 2:
            prompt = '@@ПЕРВЫЙ@@' + message + '@@ВТОРОЙ@@'
        else:
            prompt = '@@ПЕРВЫЙ@@' + self.history[-2].split(':', 1)[1].strip() + '@@ВТОРОЙ@@' + self.history[-1].split(':', 1)[1].strip() + '@@ПЕРВЫЙ@@' + message + '@@ВТОРОЙ@@'
        print('-'*20)
        print('Prompt: ' + prompt)
        inputs = self.tokenizer([prompt], return_tensors='pt')
        reply_ids = self.model.generate(
            **inputs,
            top_k=10,
            top_p=0.95,
            num_beams=3,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=1.2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40,
            pad_token_id=self.tokenizer.eos_token_id
        )
        answer = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        print('-'*20)
        print(self.char_name + ': ' + answer)
        print('-'*20)
        answer = answer[len(prompt):]
        if (idx := answer.find('@@')) != -1:
            answer = answer[:idx]
        answer = self.char_name + ': ' + answer
        self.history.append('You: ' + message)
        self.history.append(answer)
        if self.history_file_name is not None:
            try:
                with open(self.history_file_name, 'a') as f:
                    f.write('You: ' + message + '\n')
                    f.write(answer + '\n')
            except Exception:
                pass
        return answer

    def getAnswerPygmalion(self, message):
        prompt = build_prompt_for(history=self.history, 
                                  user_message=message, 
                                  char_name=self.char_name, 
                                  char_persona=self.char_persona,
                                  example_dialogue=self.example_dialogue)
        inputs = self.tokenizer([prompt], return_tensors='pt')

        # Atrocious code to stop generation when the model outputs "\nYou: " in
        # freshly generated text. Feel free to send in a PR if you know of a
        # cleaner way to do this.
        stopping_criteria_list = StoppingCriteriaList([
            _SentinelTokenStoppingCriteria(
                sentinel_token_ids=self.tokenizer(
                    "\nYou:",
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids,
                starting_idx=inputs.input_ids.shape[-1])
        ])

        reply_ids = self.model.generate(stopping_criteria=stopping_criteria_list,
                                   **inputs, 
                                   num_return_sequences=1, 
                                   max_length=512,
                                   do_sample=True,
                                   top_k=50,
                                   top_p=0.725,
                                   temperature=0.72,
                                   eos_token_id=198, 
                                   pad_token_id=self.tokenizer.eos_token_id
                               )
        answer = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        print('-'*20)
        print(answer)
        print('-'*20)
        if (idx := prompt.rfind(message)) > 0:
            answer = answer[idx + len(message) + 1:]
        if (you := answer.find('\nYou:')) > 0:
            answer = answer[:you]
        self.history.append('You: ' + message)
        self.history.append(answer)
        if self.history_file_name is not None:
            try:
                with open(self.history_file_name, 'a') as f:
                    f.write('You: ' + message + '\n')
                    f.write(answer + '\n')
            except Exception:
                pass
        return answer

    def load_config_data(self):
        try:
            with open(self.char_persona_file_name) as f:
                self.char_persona = f.read().strip()
                print(self.char_name + '\'s persona loaded')
        except Exception:
            self.char_persona = ''
            print('No persona data loaded')
        try:
            with open(self.example_dialogue_file_name) as f:
                self.example_dialogue = f.read().strip()
                print('Example dialogue loaded')
        except Exception:
            self.example_dialogue = ''
            print('No example dialogue loaded')
        try:
            with open(self.history_file_name) as f:
                for line in f:
                    if line.startswith('You: ') or line.startswith(self.char_name + ': '):
                        self.history.append(line.rstrip())
                    else:
                        self.history[-1] += '\n' + line.rstrip()
            print('Chat history loaded')
        except Exception:
            if self.bot_language == 'ru':
                self.history = [self.char_name + ': Привет!']
            else:
                self.history = [self.char_name + ': Hi!']
            print('No chat history loaded')
