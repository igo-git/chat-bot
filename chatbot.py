from ann_bot import AnnBot
from termcolor import colored

bot = AnnBot()

print("Type \"\\q\" to quit")
while True:
    message = input(colored("YOU: ", 'light_blue'))
    if '\\quit'.startswith(message) or message == '':
        break
    elif message.startswith('\\'):
        command = message.split(' ')
        if '\\voice'.startswith(command[0]):
            if command[1] == 'on':
                bot.need_voice = True
            elif command[1] == 'off':
                bot.need_voice = False
            else:
                bot.need_voice = not bot.need_voice
            print('Voice turned on' if bot.need_voice else 'Voice turned off')
        elif '\\translation'.startswith(command[0]):
            if command[1] == 'on':
                bot.need_translation = True
            elif command[1] == 'off':
                bot.need_translation = False
            else:
                bot.need_translation = not bot.need_translation
            print('Translation turned on' if bot.need_translation else 'Translation turned off')
        elif '\\reload_data'.startswith(command[0]):
            bot.load_config_data()
        else:
            print('Unrecognized command: ' + message)
        continue
#    if bot.need_translation:
#        message = translate(message, source_language='ru', target_language='en')
#        print(colored("YOU: ", 'light-blue') + message)

    answer = bot.getAnswer(message)
#    print('---\n', answer, '\n---')
    print(colored(bot.char_name, 'light_green'), answer.split(':', 1)[1])
#    if bot.need_translation:
#        answer = translate(answer, source_language='en', target_language='ru')
#        print(colored(char_name, 'light_green'), answer.split(':', 1)[1])
#    if bot.need_voice:
#        say_text(answer)