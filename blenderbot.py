from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch
import sys
import yaml
from yaml.loader import SafeLoader

def initializeModel(model_name):
    print('Initialization of ' + model_name + ' bot.')
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    print('Bot ' + model_name + ' initialized.')
    return model, tokenizer

class ChatBot():
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
        self.getAnswer = self.getAnswerBlenderbot

    def getFormattedHistory(self):
        hist = ''
        for line in self.history:
            hist += '***' + line.split(':', 1)[0] + ':*** ' + line.split(':', 1)[1] + '\n\n'
        return hist

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
