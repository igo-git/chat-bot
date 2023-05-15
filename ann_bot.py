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
        try:
            with open('config.yaml') as file:
                config = yaml.load(file, Loader=SafeLoader)
        except Exception:
            print('Cann\'t read config file')
            exit(0)
        self.history = []
        self.char_name = config['chat_bot']['char_name']
        self.char_persona_file_name = config['chat_bot']['char_persona_file_name']
        self.example_dialogue_file_name = config['chat_bot']['example_dialogue_file_name']
        self.history_file_name = config['chat_bot']['history_file_name']
        self.load_config_data()
        self.model_name = config['chat_bot']['model_name']
        self.model, self.tokenizer = initializeModel(self.model_name)
        self.need_translation = config['chat_bot']['need_translation']
        self.need_voice = config['chat_bot']['need_voice']

        self.chat_history_ids = torch.zeros((1, 0), dtype=torch.int)

        if self.model is None:
            print('No model initialized')
            sys.exit(0)

    def getAnswer(self, message):
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
        with open(self.history_file_name, 'a') as f:
            f.write('You: ' + message + '\n')
            f.write(answer + '\n')
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
            self.history = ['Ann: Hi!']
            print('No chat history loaded')
