#!/usr/bin/env python

from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit
from PyQt6.QtGui import QTextCursor
#from tts import sayText
import sys
from time import sleep
import yaml
from yaml.loader import SafeLoader
from importlib import import_module

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("QChatBot - " + bot.char_name)
        self.chatHistoryViewer = QTextEdit(readOnly=True)
        self.chatHistoryViewer.setMarkdown(bot.getFormattedHistory())
        self.chatHistoryViewer.verticalScrollBar().setValue(self.chatHistoryViewer.verticalScrollBar().maximum())
#        self.chatHistoryViewer.setFontPointSize(12)
        self.userMessageEditor = QTextEdit()
#        self.userMessageEditor.setFontPointSize(12)
        self.sayButton = QPushButton('Say')
        self.sayButton.setAutoDefault(True)
        self.sayButton.clicked.connect(self.sayButton_clicked)
        self.sayButton.setShortcut('Ctrl+Return')

        layout = QVBoxLayout()
        layout.addWidget(self.chatHistoryViewer)
        layout.addWidget(self.userMessageEditor)
        layout.addWidget(self.sayButton)

        container = QWidget()
        container.setLayout(layout)

        # Устанавливаем центральный виджет Window.
        self.setCentralWidget(container)
        self.userMessageEditor.setFocus()

    def sayButton_clicked(self):
        message = self.userMessageEditor.toPlainText().strip()
        if message != '':
            if '\\quit'.startswith(message):
                QApplication.instance().quit()
            elif message.startswith('\\'):
                command = message.split(' ')
                if '\\voice'.startswith(command[0]):
                    if len(command) == 1:
                        bot.need_voice = not bot.need_voice
                    elif command[1] == 'on':
                        bot.need_voice = True
                    elif command[1] == 'off':
                        bot.need_voice = False
                    else:
                        bot.need_voice = not bot.need_voice
#                    sayText('Voice turned on' if bot.need_voice else 'Voice turned off')
                    self.userMessageEditor.setText('')
            else:    
                hist = self.chatHistoryViewer.toMarkdown() + '\n\n***You:***  ' + message
                self.chatHistoryViewer.setMarkdown(hist)
                self.chatHistoryViewer.verticalScrollBar().setValue(self.chatHistoryViewer.verticalScrollBar().maximum())
                self.userMessageEditor.setText('')
                self.sayButton.setEnabled(False)
                QApplication.processEvents()
                answer = bot.getAnswer(message)
                hist = self.chatHistoryViewer.toMarkdown() + '***' + bot.char_name + ':*** ' + answer.split(':', 1)[1]
                self.chatHistoryViewer.setMarkdown(hist)
                self.chatHistoryViewer.verticalScrollBar().setValue(self.chatHistoryViewer.verticalScrollBar().maximum())
                QApplication.processEvents()
                # if bot.need_voice:
                #     sayText(answer.split(':', 1)[1], bot.bot_language)
                self.sayButton.setEnabled(True)
                self.userMessageEditor.setFocus()

if len(sys.argv) > 1:
    config_file_name = sys.argv[1]
else:
    config_file_name = 'config.yaml'
try:
    with open(config_file_name) as file:
        config = yaml.load(file, Loader=SafeLoader)
except Exception:
    print('Cann\'t read config file ' + config_file_name)
    exit(0)
if 'bot_module' not in config['chat_bot'].keys():
    print('No bot module specified')
    exit(0)
bot_module = import_module(config['chat_bot']['bot_module'])

bot = bot_module.ChatBot(config_file_name)
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
