import openai
import yaml
from yaml.loader import SafeLoader

class ChatBot():
    def __init__(self, config_file_name=''):
        self.first_step = True
        self.config_file_name = config_file_name
        try:
            with open(self.config_file_name) as file:
                config = yaml.load(file, Loader=SafeLoader)
        except Exception:
            print('gpt_bot.py error: Cann\'t read config file ' + config_file_name)
            exit(0)
        if 'openai_api_key' in config['chat_bot'].keys():
            openai.api_key = config['chat_bot']['openai_api_key']
        else:
            print('No openai API key specified')
            exit(0)
        if 'char_name' in config['chat_bot'].keys():
            self.char_name = config['chat_bot']['char_name']
        else:
            self.char_name = 'ChatGPT'
        if 'model' in config['chat_bot'].keys():
            self.model_name = config['chat_bot']['model_name']
        else:
            self.model_name = 'gpt-3.5-turbo'
        self.history = []

    def getFormattedHistory(self):
        hist = ''
        for line in self.history:
            hist += '***' + line.split(':', 1)[0] + ':*** ' + line.split(':', 1)[1] + '\n\n'
        return hist


    def getAnswer(self, message):
        answer = openai.ChatCompletion.create(model=self.model_name, messages=[{'role':'user', 'content':message}])
        return answer

        
